from collections.abc import AsyncIterator

from .base import LLMProvider


class OpenAILLM(LLMProvider):
    """Streaming LLM provider backed by the OpenAI chat completions API."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self.api_key = api_key
        self.model = model
        # TODO: Issue #5 — initialize the AsyncOpenAI client
        # self.client = AsyncOpenAI(api_key=api_key)

    async def generate(self, messages: list[dict]) -> AsyncIterator[str]:
        # TODO: Issue #5 — stream chat completion tokens and yield each chunk
        raise NotImplementedError
