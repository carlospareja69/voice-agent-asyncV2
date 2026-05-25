import asyncio
import logging

import sounddevice

logger = logging.getLogger(__name__)


class MicrophoneInput:
    """Captures audio from the default microphone and pushes chunks to a queue.

    Audio format
    ------------
    dtype   : float32
    channels: 1 (mono)
    layout  : each chunk is a raw bytes object carrying `chunk_size` float32
              samples.  The STT stage reconstructs the numpy array with:

                  import numpy as np
                  audio = np.frombuffer(chunk, dtype=np.float32)

    Threading model
    ---------------
    sounddevice fires its callback on a dedicated PortAudio audio thread, not
    on the asyncio event loop thread.  Calling asyncio primitives directly from
    that thread is not safe.  We bridge the two worlds using:

        loop.call_soon_threadsafe(queue.put_nowait, audio_bytes)

    This posts the put_nowait call onto the event loop's internal pipe so the
    loop executes it safely from its own thread on the next iteration.

    This class knows nothing about the pipeline internals — it only writes to
    the queue it is given.
    """

    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024) -> None:
        """
        Args:
            sample_rate: Capture rate in Hz.  Must match the STT model's
                         expected input rate — faster-whisper expects 16 000 Hz.
            chunk_size:  Number of audio frames delivered per callback.
                         Smaller values lower latency; larger values lower CPU
                         overhead from frequent callbacks.
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

    async def stream(self, queue: asyncio.Queue[bytes]) -> None:
        """Open the default microphone and push audio chunks into *queue*.

        Runs indefinitely until the calling task is cancelled.  On cancellation
        the sounddevice stream is closed cleanly by the context manager and
        CancelledError is re-raised so the pipeline task group can detect the
        shutdown.

        Args:
            queue: Destination queue.  Each item is a bytes object containing
                   `chunk_size` float32 PCM samples for the STT stage.

        Raises:
            sounddevice.PortAudioError: if no input device is available.
            asyncio.CancelledError: always re-raised on task cancellation.
        """
        # Capture the running loop here, in the event loop thread, so the
        # callback closure can reference it safely from the audio thread.
        loop = asyncio.get_running_loop()

        def _callback(
            indata,       # numpy float32 array — shape (chunk_size, channels)
            frames: int,  # number of frames delivered (equals chunk_size)
            time,         # sounddevice CData timing info — unused
            status,       # sounddevice CallbackFlags
        ) -> None:
            """Called by PortAudio on a background audio thread."""
            if status:
                # e.g. input_overflow means the app is not consuming fast enough
                logger.warning("Microphone status flag: %s", status)

            # .tobytes() creates an independent bytes copy of the numpy buffer.
            # This is essential: `indata` is a view into a PortAudio-managed
            # buffer that will be overwritten on the very next callback.
            audio_bytes = indata.tobytes()

            # Bridge from the audio thread to the event loop thread.
            # put_nowait is safe to call here because call_soon_threadsafe
            # schedules it on the loop's own thread via the internal wakeup pipe.
            loop.call_soon_threadsafe(queue.put_nowait, audio_bytes)

        with sounddevice.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            channels=1,
            dtype="float32",
            callback=_callback,
        ):
            logger.info(
                "Microphone open — sample_rate=%d Hz, chunk_size=%d frames",
                self.sample_rate,
                self.chunk_size,
            )
            try:
                # Hold this coroutine alive so the context manager keeps the
                # stream open.  asyncio.Event().wait() suspends without polling
                # and exits only when the task is cancelled.
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                logger.info("Microphone input stopped.")
                raise  # propagate — do not swallow CancelledError
