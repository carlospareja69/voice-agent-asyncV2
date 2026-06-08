import logging
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from .base import LLMProvider

logger = logging.getLogger(__name__)


class OpenAILLM(LLMProvider):
    """Streaming LLM provider backed by the OpenAI chat completions API.

    Uses the openai SDK v1+ AsyncOpenAI client.  Each call to generate()
    opens a server-sent-events stream, yields decoded text tokens as they
    arrive, and closes the HTTP connection in a try/finally block whether
    iteration completes normally, is interrupted by a break, or is cancelled.

    Usage
    -----
    The method is an async generator — callers iterate with ``async for``:

        async for token in llm.generate(messages):
            print(token, end="", flush=True)

    Do NOT await the call — there is nothing to await.  The first token
    arrives after the server begins streaming; subsequent tokens arrive as
    the model produces them.

    Args
    ----
    api_key : str
        OpenAI API key.  Passed to AsyncOpenAI at construction time.
    model : str
        OpenAI model identifier.  Defaults to "gpt-4o-mini".
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key)

    async def generate(self, messages: list[dict]) -> AsyncIterator[str]:
        """Stream chat completion tokens for the given message history.

        Opens a streaming request to the OpenAI chat completions endpoint.
        Each non-None delta content string is yielded to the caller.  Empty
        choices chunks (keep-alive frames) and None deltas are silently
        skipped.

        The underlying HTTP stream is always closed in the finally block —
        even if the caller breaks early, the task is cancelled, or an
        exception propagates.

        Args:
            messages: List of dicts with "role" and "content" keys,
                      in the format expected by the OpenAI API.

        Yields:
            Individual text tokens (str) as they arrive from the API.
        """
        logger.debug(
            "Starting LLM stream: model=%s, messages=%d", self.model, len(messages)
        )

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
        )

        try:
            async for chunk in stream:
                if not chunk.choices:
                    # Keep-alive frame sent by the API — skip it.
                    continue
                content = chunk.choices[0].delta.content
                if content is not None:
                    yield content
        finally:
            await stream.close()
            logger.debug("LLM stream closed.")
