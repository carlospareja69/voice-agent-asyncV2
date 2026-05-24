from collections.abc import AsyncIterator

from .base import TTSProvider


class ElevenLabsTTS(TTSProvider):
    """Streaming TTS provider backed by the ElevenLabs HTTP API."""

    def __init__(self, api_key: str, voice_id: str) -> None:
        self.api_key = api_key
        self.voice_id = voice_id
        # TODO: Issue #6 — initialize an aiohttp.ClientSession here

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        # TODO: Issue #6 — POST to ElevenLabs streaming endpoint and yield audio chunks
        raise NotImplementedError
