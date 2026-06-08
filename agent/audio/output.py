"""Async speaker output stage.

Reads raw PCM audio bytes from an ``asyncio.Queue`` and plays them through
the default audio output device via ``sounddevice.OutputStream``.

Threading model
---------------
``sounddevice`` (PortAudio) fires the output callback on a real-time audio
thread — not the asyncio event loop thread.  ``asyncio.Queue`` is **not**
thread-safe for use from that callback.

The solution is a two-layer buffer:

1. The asyncio task reads chunks from the asyncio queue and puts them into a
   stdlib ``queue.Queue`` (thread-safe).
2. The PortAudio callback reads from the stdlib queue with ``get_nowait()``
   (non-blocking — never blocks the audio thread).

Chunk/frame alignment
---------------------
ElevenLabs TTS produces chunks of a fixed byte size (``_CHUNK_SIZE = 4096``
bytes ≈ 2048 int16 samples), but PortAudio requests ``frames`` samples per
callback invocation (platform-dependent, commonly 256–2048).  These sizes
do not align.

A ``bytearray`` leftover accumulator inside the callback closure bridges the
mismatch: it accumulates from the stdlib queue until enough bytes are
available for the current callback request, then fills ``outdata`` and
retains any remainder for the next callback.

Audio format contract (tts_queue)
----------------------------------
- Encoding   : signed 16-bit integer (int16), little-endian
- Sample rate : 24 000 Hz  (matches ElevenLabsTTS ``output_format="pcm_24000"``)
- Channels   : 1 (mono)

Changing ``sample_rate`` without updating the ElevenLabsTTS constructor call
and the ``TTS_SAMPLE_RATE`` env var will produce garbled audio.
"""

import asyncio
import logging
import queue as _queue

import numpy as np
import sounddevice

logger = logging.getLogger(__name__)


class SpeakerOutput:
    """Reads audio bytes from a queue and plays them through the default speaker.

    Args:
        sample_rate: Output sample rate in Hz.  Must match the encoding of
            bytes arriving on the queue.  Defaults to 24 000 Hz, which
            corresponds to ElevenLabs ``output_format="pcm_24000"``.
    """

    def __init__(self, sample_rate: int = 24000) -> None:
        self.sample_rate = sample_rate

    async def stream(self, queue: asyncio.Queue[bytes]) -> None:
        """Open a speaker stream and drain ``queue`` into it until cancelled.

        Opens a ``sounddevice.OutputStream`` for int16 mono playback at
        ``self.sample_rate`` Hz.  An asyncio loop reads raw PCM bytes from
        ``queue`` and forwards them to a thread-safe stdlib ``queue.Queue``
        that the PortAudio callback reads from.

        The coroutine runs until an ``asyncio.CancelledError`` is raised
        (e.g. when the pipeline task is cancelled), at which point the
        ``OutputStream`` context manager is exited and the error is
        re-raised so callers see a clean shutdown.

        Args:
            queue: Source of raw PCM bytes.  Each item must be a ``bytes``
                object containing signed int16 samples at ``self.sample_rate``
                Hz, mono.

        Raises:
            asyncio.CancelledError: Propagated to the caller on task
                cancellation so the pipeline can shut down cleanly.
        """
        # Thread-safe bridge: asyncio task → PortAudio callback thread.
        _buffer: _queue.Queue[bytes] = _queue.Queue()

        # Accumulates unconsumed bytes between callback invocations so that
        # chunk/frame size misalignment is handled transparently.
        leftover = bytearray()

        def callback(
            outdata: np.ndarray,
            frames: int,
            time,  # noqa: ANN001 — cffi time struct, not used
            status,
        ) -> None:
            """PortAudio output callback — runs on the audio thread."""
            if status:
                logger.warning("Speaker status flag: %s", status)

            needed_bytes = frames * 2  # 1 channel × 2 bytes per int16 sample

            # Drain the thread-safe queue into leftover until we have enough
            # bytes for this callback, or the queue is exhausted.
            while len(leftover) < needed_bytes:
                try:
                    leftover.extend(_buffer.get_nowait())
                except _queue.Empty:
                    break

            available_frames = min(len(leftover) // 2, frames)

            if available_frames > 0:
                outdata[:available_frames, 0] = np.frombuffer(
                    leftover[: available_frames * 2], dtype=np.int16
                )
                del leftover[: available_frames * 2]

            # Pad with silence if there is not enough audio data.
            if available_frames < frames:
                outdata[available_frames:, 0] = 0

        with sounddevice.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            callback=callback,
        ):
            logger.info("Speaker open — sample_rate=%d Hz", self.sample_rate)
            try:
                while True:
                    chunk = await queue.get()
                    _buffer.put_nowait(chunk)
            except asyncio.CancelledError:
                logger.info("Speaker output stopped.")
                raise
