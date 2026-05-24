from abc import ABC, abstractmethod


class STTProvider(ABC):
    """Abstract interface for speech-to-text providers."""

    @abstractmethod
    async def transcribe(self, audio: bytes) -> str:
        """Convert raw PCM audio bytes into a text transcript."""
        ...
