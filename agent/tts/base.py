from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class TTSProvider(ABC):
    """Abstract interface for text-to-speech providers."""

    @abstractmethod
    def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Stream audio bytes from a text string."""
        ...
