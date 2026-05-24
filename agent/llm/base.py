from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class LLMProvider(ABC):
    """Abstract interface for large language model providers."""

    @abstractmethod
    async def generate(self, messages: list[dict]) -> AsyncIterator[str]:
        """Stream text tokens given a list of chat messages."""
        ...
