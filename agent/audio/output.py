import asyncio


class SpeakerOutput:
    """Reads audio bytes from a queue and plays them through the default speaker."""

    def __init__(self, sample_rate: int = 24000) -> None:
        self.sample_rate = sample_rate
        # TODO: Issue #7 — configure sounddevice OutputStream

    async def stream(self, queue: asyncio.Queue) -> None:
        """Continuously dequeue audio bytes and write them to the speaker."""
        # TODO: Issue #7 — open sounddevice output stream, drain queue in a loop
        raise NotImplementedError
