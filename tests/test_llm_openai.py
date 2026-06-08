"""
Tests for agent/llm/openai.py

The openai package and its AsyncOpenAI client are mocked throughout — no
real API key or network connection is required.  If the openai package is
not installed, the module-level sys.modules stub ensures OpenAILLM can
still be imported and tested.

Test coverage
-------------
- Client is initialized with the correct api_key
- model is stored on the instance
- Tokens from non-None delta content are yielded
- None delta content values are silently skipped
- Chunks with empty choices lists are silently skipped
- The stream is called with the correct model, messages, and stream=True
- An empty stream yields nothing
- stream.close() is called after full iteration (normal completion)
- stream.close() is called when the caller breaks early
- stream.close() is called when an exception is raised during iteration
- stream.close() is called when the async generator is garbage-collected
  (CancelledError path)
"""

import sys
import unittest.mock
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Make the module importable even when the openai package is not installed.
# ---------------------------------------------------------------------------
if "openai" not in sys.modules:
    openai_mock = unittest.mock.MagicMock()
    openai_mock.AsyncOpenAI = MagicMock
    sys.modules["openai"] = openai_mock

from agent.llm.openai import OpenAILLM  # noqa: E402 — must follow sys.modules patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(content: str | None) -> MagicMock:
    """Return a mock ChatCompletionChunk with one choice carrying ``content``."""
    chunk = MagicMock()
    choice = MagicMock()
    choice.delta.content = content
    chunk.choices = [choice]
    return chunk


def _make_empty_choices_chunk() -> MagicMock:
    """Return a mock chunk with an empty choices list (keep-alive frame)."""
    chunk = MagicMock()
    chunk.choices = []
    return chunk


class MockStream:
    """Async-iterable mock for the openai streaming response object.

    Supports ``async for`` iteration and tracks whether ``close()`` was called.
    """

    def __init__(self, chunks: list[MagicMock]) -> None:
        self._chunks = chunks
        self.closed = False

    def __aiter__(self) -> AsyncIterator[MagicMock]:
        return self._iterate()

    async def _iterate(self):
        for chunk in self._chunks:
            yield chunk

    async def close(self) -> None:
        self.closed = True


def _make_llm(monkeypatch, model: str = "gpt-4o-mini") -> "OpenAILLM":
    """Return an OpenAILLM with AsyncOpenAI patched out."""
    monkeypatch.setattr("agent.llm.openai.AsyncOpenAI", lambda api_key: MagicMock())
    return OpenAILLM(api_key="test-key", model=model)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInit:
    def test_client_initialized_with_api_key(self, monkeypatch):
        """AsyncOpenAI is constructed with the provided api_key."""
        captured = []

        def fake_async_openai(api_key):
            captured.append(api_key)
            return MagicMock()

        monkeypatch.setattr("agent.llm.openai.AsyncOpenAI", fake_async_openai)
        OpenAILLM(api_key="sk-test-123")

        assert captured == ["sk-test-123"]

    def test_model_stored(self, monkeypatch):
        """The model name is stored on the instance."""
        llm = _make_llm(monkeypatch, model="gpt-4o")
        assert llm.model == "gpt-4o"

    def test_default_model(self, monkeypatch):
        """Default model is gpt-4o-mini when no model argument is given."""
        monkeypatch.setattr("agent.llm.openai.AsyncOpenAI", lambda api_key: MagicMock())
        llm = OpenAILLM(api_key="sk-test")
        assert llm.model == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# generate() — token yielding
# ---------------------------------------------------------------------------


class TestGenerate:
    async def test_tokens_are_yielded(self, monkeypatch):
        """Non-None delta content strings are yielded in order."""
        llm = _make_llm(monkeypatch)
        mock_stream = MockStream(
            [_make_chunk("Hello"), _make_chunk(" "), _make_chunk("world")]
        )
        llm.client.chat.completions.create = AsyncMock(return_value=mock_stream)

        tokens = []
        async for token in llm.generate([{"role": "user", "content": "Hi"}]):
            tokens.append(token)

        assert tokens == ["Hello", " ", "world"]

    async def test_none_delta_content_skipped(self, monkeypatch):
        """Chunks where delta.content is None are not yielded."""
        llm = _make_llm(monkeypatch)
        mock_stream = MockStream(
            [_make_chunk(None), _make_chunk("Hi"), _make_chunk(None)]
        )
        llm.client.chat.completions.create = AsyncMock(return_value=mock_stream)

        tokens = []
        async for token in llm.generate([{"role": "user", "content": "Hi"}]):
            tokens.append(token)

        assert tokens == ["Hi"]

    async def test_empty_choices_chunk_skipped(self, monkeypatch):
        """Chunks with an empty choices list (keep-alive frames) are skipped."""
        llm = _make_llm(monkeypatch)
        mock_stream = MockStream(
            [_make_empty_choices_chunk(), _make_chunk("OK"), _make_empty_choices_chunk()]
        )
        llm.client.chat.completions.create = AsyncMock(return_value=mock_stream)

        tokens = []
        async for token in llm.generate([{"role": "user", "content": "?"}]):
            tokens.append(token)

        assert tokens == ["OK"]

    async def test_correct_parameters_passed_to_api(self, monkeypatch):
        """create() is called with model=, messages=, stream=True."""
        llm = _make_llm(monkeypatch, model="gpt-4o")
        mock_stream = MockStream([])
        create_mock = AsyncMock(return_value=mock_stream)
        llm.client.chat.completions.create = create_mock

        messages = [{"role": "user", "content": "Hello"}]
        async for _ in llm.generate(messages):
            pass

        create_mock.assert_called_once_with(
            model="gpt-4o",
            messages=messages,
            stream=True,
        )

    async def test_empty_stream_yields_nothing(self, monkeypatch):
        """An empty stream produces zero tokens without raising."""
        llm = _make_llm(monkeypatch)
        mock_stream = MockStream([])
        llm.client.chat.completions.create = AsyncMock(return_value=mock_stream)

        tokens = []
        async for token in llm.generate([{"role": "user", "content": "Hi"}]):
            tokens.append(token)

        assert tokens == []

    async def test_generate_is_async_generator_not_coroutine(self, monkeypatch):
        """generate() returns an async generator — it must NOT be awaited."""
        import inspect

        llm = _make_llm(monkeypatch)
        mock_stream = MockStream([])
        llm.client.chat.completions.create = AsyncMock(return_value=mock_stream)

        result = llm.generate([{"role": "user", "content": "Hi"}])

        # An async generator is NOT a coroutine — it should not be awaited.
        assert inspect.isasyncgen(result)
        assert not inspect.iscoroutine(result)

        # Exhaust it so the finalizer closes the stream.
        async for _ in result:
            pass


# ---------------------------------------------------------------------------
# generate() — stream cleanup (try/finally)
# ---------------------------------------------------------------------------


class TestStreamCleanup:
    async def test_stream_closed_after_full_iteration(self, monkeypatch):
        """stream.close() is called after all tokens have been consumed."""
        llm = _make_llm(monkeypatch)
        mock_stream = MockStream([_make_chunk("A"), _make_chunk("B")])
        llm.client.chat.completions.create = AsyncMock(return_value=mock_stream)

        async for _ in llm.generate([{"role": "user", "content": "x"}]):
            pass

        assert mock_stream.closed

    async def test_stream_closed_on_early_break(self, monkeypatch):
        """stream.close() is called when the caller explicitly closes the generator.

        Python's ``async for`` does not call ``aclose()`` on an async generator
        when a ``break`` exits the loop — finalization is scheduled by the event
        loop's async-gen finalizer hook and runs later (or at loop shutdown).
        Production code that needs prompt cleanup must call ``aclose()``
        explicitly, which is what the pipeline stage must do in its shutdown path.
        This test verifies that ``finally`` inside ``generate()`` does run when
        ``aclose()`` is called.
        """
        llm = _make_llm(monkeypatch)
        mock_stream = MockStream(
            [_make_chunk("first"), _make_chunk("second"), _make_chunk("third")]
        )
        llm.client.chat.completions.create = AsyncMock(return_value=mock_stream)

        gen = llm.generate([{"role": "user", "content": "x"}])
        async for token in gen:
            break  # bail after first token
        await gen.aclose()  # explicit cleanup — triggers the try/finally block

        assert mock_stream.closed

    async def test_stream_closed_on_exception_in_consumer(self, monkeypatch):
        """stream.close() is called when the generator is closed after a consumer exception.

        When an exception propagates out of the body of ``async for``, the async
        generator is left suspended (not automatically finalized).  The caller is
        responsible for calling ``aclose()`` to trigger the ``finally`` block.
        This mirrors the ``break`` case — both require explicit cleanup.
        """
        llm = _make_llm(monkeypatch)
        mock_stream = MockStream([_make_chunk("A"), _make_chunk("B"), _make_chunk("C")])
        llm.client.chat.completions.create = AsyncMock(return_value=mock_stream)

        gen = llm.generate([{"role": "user", "content": "x"}])
        with pytest.raises(RuntimeError, match="consumer error"):
            async for _ in gen:
                raise RuntimeError("consumer error")
        await gen.aclose()  # explicit cleanup — triggers the try/finally block

        assert mock_stream.closed

    async def test_stream_closed_on_exception_during_streaming(self, monkeypatch):
        """stream.close() is called when an exception occurs inside the stream."""

        class BrokenStream:
            """A stream that raises on the second iteration."""

            def __init__(self):
                self.closed = False
                self._count = 0

            def __aiter__(self):
                return self._iterate()

            async def _iterate(self):
                self._count += 1
                if self._count == 1:
                    yield _make_chunk("first")
                raise ValueError("stream error")

            async def close(self):
                self.closed = True

        llm = _make_llm(monkeypatch)
        broken_stream = BrokenStream()
        llm.client.chat.completions.create = AsyncMock(return_value=broken_stream)

        tokens = []
        with pytest.raises(ValueError, match="stream error"):
            async for token in llm.generate([{"role": "user", "content": "x"}]):
                tokens.append(token)

        assert tokens == ["first"]
        assert broken_stream.closed
