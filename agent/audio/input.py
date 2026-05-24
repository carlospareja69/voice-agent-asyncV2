import asyncio


class MicrophoneInput:
    """Captures audio from the default microphone and pushes chunks to a queue."""

    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024) -> None:
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        # TODO: Issue #3 — configure sounddevice InputStream

    async def stream(self, queue: asyncio.Queue) -> None:
        """Continuously read mic chunks and put raw PCM bytes into queue."""
        # TODO: Issue #3 — open sounddevice stream, read in a loop, enqueue chunks
        raise NotImplementedError
