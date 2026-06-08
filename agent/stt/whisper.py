import asyncio
import logging

import numpy as np
from faster_whisper import WhisperModel

from .base import STTProvider

logger = logging.getLogger(__name__)


class WhisperSTT(STTProvider):
    """Speech-to-text provider using faster-whisper running locally on CPU.

    The model is loaded once at construction time (synchronous, expensive) and
    reused for every transcription call.  All inference runs inside a thread-pool
    executor so the asyncio event loop is never blocked.

    Audio format expected by transcribe()
    --------------------------------------
    The pipeline sends audio as raw bytes produced by MicrophoneInput:
        - dtype   : float32
        - channels: 1 (mono)
        - rate    : must match the sample_rate this instance was created with
                    (defaults to 16 000 Hz — what Whisper expects)

    Reconstruct the numpy array with:
        np.frombuffer(audio_bytes, dtype=np.float32)
    """

    def __init__(
        self,
        model_name: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        sample_rate: int = 16000,
    ) -> None:
        """Load the Whisper model.

        This is intentionally synchronous and called before asyncio.run() so
        the event loop is never blocked by model loading.

        Args:
            model_name:   faster-whisper model size.  Must be one of the sizes
                          validated by Settings (tiny, base, small, …).
            device:       Inference device — "cpu" or "cuda".
            compute_type: Quantisation level — "int8" (fast, small) or
                          "float16" / "float32" (higher quality).
            sample_rate:  Expected audio sample rate in Hz.  Must match the
                          rate used by MicrophoneInput (Settings.audio_sample_rate).
        """
        self.model_name = model_name
        self.sample_rate = sample_rate

        logger.info(
            "Loading Whisper model '%s' on %s (%s) …",
            model_name,
            device,
            compute_type,
        )
        self._model = WhisperModel(model_name, device=device, compute_type=compute_type)
        logger.info("Whisper model ready.")

    async def transcribe(self, audio: bytes) -> str:
        """Transcribe raw PCM audio bytes into a text string.

        The audio must be float32 mono PCM at self.sample_rate Hz — exactly the
        format produced by MicrophoneInput and stored in audio_queue.

        Inference runs in a thread-pool executor (faster-whisper is synchronous
        and CPU-bound).  The event loop is free to serve other tasks while
        transcription is in progress.

        Args:
            audio: Raw float32 PCM bytes from audio_queue.

        Returns:
            Stripped transcript text, or "" if the audio contained only silence
            or was too short to produce a result.
        """
        # Reconstruct the numpy array.  np.frombuffer returns a read-only view
        # of the bytes object — no copy needed; faster-whisper only reads it.
        audio_array = np.frombuffer(audio, dtype=np.float32)

        def _sync_transcribe() -> str:
            # model.transcribe() returns a lazy generator of segments.
            # We consume it fully here, inside the executor thread, so the
            # event loop receives a plain str — not an unconsumed generator.
            segments, _ = self._model.transcribe(audio_array, beam_size=5)
            return " ".join(segment.text for segment in segments).strip()

        loop = asyncio.get_running_loop()
        transcript = await loop.run_in_executor(None, _sync_transcribe)

        if transcript:
            logger.debug("Transcript: %r", transcript)
        else:
            logger.debug("Transcription produced no text (silence or short audio).")

        return transcript
