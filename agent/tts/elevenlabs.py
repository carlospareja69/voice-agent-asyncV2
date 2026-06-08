import logging
from collections.abc import AsyncIterator

import aiohttp

from .base import TTSProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_BASE_URL = "https://api.elevenlabs.io/v1/text-to-speech"

# Number of bytes read per iteration from the HTTP response body.
# At pcm_24000 (int16 mono), 4096 bytes ≈ 85 ms of audio — low enough for
# responsive streaming, large enough to avoid excessive loop overhead.
_CHUNK_SIZE = 4096


class ElevenLabsTTS(TTSProvider):
    """Streaming TTS provider backed by the ElevenLabs HTTP API.

    Each call to ``synthesize()`` opens a new ``aiohttp.ClientSession``,
    POSTs the text to the ElevenLabs streaming endpoint, and yields raw
    audio bytes as they arrive over the HTTP response body.  The session
    and connection are closed when the generator is fully exhausted or
    explicitly finalised via ``aclose()``.

    Audio format contract
    ---------------------
    By default this class requests ``output_format="pcm_24000"`` from
    ElevenLabs.  Every ``bytes`` object yielded by ``synthesize()`` is a
    fragment of a raw PCM audio stream with the following properties:

        - Encoding : signed 16-bit integer (int16), little-endian
        - Sample rate : 24 000 Hz
        - Channels : 1 (mono)

    Downstream consumers must be configured to match this format.
    ``SpeakerOutput`` (Issue #7) must open its ``sounddevice.OutputStream``
    with ``dtype="int16"``, ``samplerate=24000``, ``channels=1``.
    ``tts_queue`` items can be reconstructed as a numpy array with::

        np.frombuffer(chunk, dtype=np.int16)

    This is the single authoritative description of ``tts_queue`` items.
    Changing ``output_format`` invalidates this contract — both the
    ElevenLabsTTS constructor call and the SpeakerOutput configuration
    must be updated together.

    Session strategy
    ----------------
    A new ``aiohttp.ClientSession`` is created for every ``synthesize()``
    call (Option C — session-per-request).  This avoids lifecycle
    management complexity: no ``close()`` method is required on this class
    and the pipeline does not need to call ``tts.close()`` at shutdown.

    For a sequential voice-agent pipeline (one TTS call at a time) the
    per-call overhead of a fresh TCP handshake is imperceptible (~5–10 ms).
    If concurrent TTS calls were ever needed, a persistent shared session
    would improve throughput.  That is not the current architecture.

    Cancellation contract
    ---------------------
    The ``finally`` clauses of ``synthesize()`` run when the generator is
    finalised.  Python's ``async for`` does **not** call ``aclose()``
    automatically when a ``break`` or consumer exception exits the loop
    body.  The pipeline's TTS stage (Issue #8) must hold a reference to
    the active generator and call ``await gen.aclose()`` before re-raising
    ``CancelledError`` to ensure the HTTP connection is released promptly.

    Args
    ----
    api_key : str
        ElevenLabs API key.  Never stored as a public attribute.
    voice_id : str
        ElevenLabs voice identifier.
    model_id : str
        ElevenLabs model name.  Defaults to ``"eleven_monolingual_v1"``.
    output_format : str
        ElevenLabs output format identifier.  Defaults to ``"pcm_24000"``.
        Changing this changes the audio format of yielded bytes — update
        ``SpeakerOutput`` and any downstream consumer to match.
    """

    def __init__(
        self,
        api_key: str,
        voice_id: str,
        model_id: str = "eleven_monolingual_v1",
        output_format: str = "pcm_24000",
    ) -> None:
        self._api_key = api_key  # private — passed in headers only, never logged
        self.voice_id = voice_id
        self.model_id = model_id
        self.output_format = output_format

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Stream raw PCM audio bytes for the given text.

        Opens a streaming POST request to the ElevenLabs TTS endpoint and
        yields audio byte fragments as they arrive.  The HTTP connection is
        closed when iteration completes normally, the caller calls
        ``aclose()``, or an exception propagates through the generator.

        Args:
            text: The text to synthesise.  Should be a complete sentence
                  or phrase — the ElevenLabs model produces better prosody
                  with sentence-level input than with individual tokens.

        Yields:
            Raw audio bytes.  Format is determined by ``self.output_format``
            (default: signed int16, 24 kHz, mono).  Empty fragments are
            silently discarded.

        Raises:
            aiohttp.ClientResponseError: If the API returns a non-2xx
                status (e.g. 401 invalid key, 422 bad voice_id, 429 rate
                limit).  The exception propagates to the caller without
                any retrying.
        """
        url = f"{_BASE_URL}/{self.voice_id}/stream"
        headers = {
            "xi-api-key": self._api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
            },
        }
        params = {"output_format": self.output_format}

        logger.debug(
            "Starting TTS stream: voice=%s, model=%s, chars=%d",
            self.voice_id,
            self.model_id,
            len(text),
        )

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=headers,
                json=payload,
                params=params,
            ) as response:
                response.raise_for_status()
                async for chunk in response.content.iter_chunked(_CHUNK_SIZE):
                    if chunk:
                        yield chunk

        logger.debug("TTS stream closed.")
