"""
Tests for agent/tts/elevenlabs.py

The aiohttp package is mocked throughout — no real API key or network
connection is required.  If the aiohttp package is not installed, the
module-level sys.modules stub ensures ElevenLabsTTS can still be imported
and tested.

Mock hierarchy
--------------
MockContent       — async generator implementing iter_chunked(n)
MockResponse      — async context manager with raise_for_status() + .content
MockSession       — async context manager whose .post() returns MockResponse
_make_session     — factory that builds a MockSession and wires it up

Test coverage
-------------
- voice_id, model_id, output_format stored on instance
- Default model_id is "eleven_monolingual_v1"
- Default output_format is "pcm_24000"
- Custom model_id and output_format accepted
- Non-empty audio chunks are yielded in order
- Empty chunks (b"") are silently discarded
- URL contains the voice_id
- xi-api-key header carries the api_key
- output_format query parameter is forwarded to the API
- text field is present in the JSON payload
- raise_for_status() is called before iteration begins
- HTTP errors propagate as exceptions without yielding
- synthesize() returns an async generator, not a coroutine
- Session and response context managers are exited after full iteration
- Session and response context managers are exited on explicit aclose()
"""

import inspect
import sys
import unittest.mock
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Make the module importable even when aiohttp is not installed.
# The stub must expose ClientSession and ClientResponseError at minimum.
# ---------------------------------------------------------------------------
if "aiohttp" not in sys.modules:
    aiohttp_mock = unittest.mock.MagicMock()
    aiohttp_mock.ClientResponseError = Exception
    sys.modules["aiohttp"] = aiohttp_mock

from agent.tts.elevenlabs import ElevenLabsTTS  # noqa: E402


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class MockContent:
    """Async generator stub for ``aiohttp.StreamReader``.

    Implements ``iter_chunked(n)`` as an async generator that yields each
    item from ``chunks`` unchanged.  Chunk size ``n`` is accepted but
    ignored — the mock yields whatever chunks it was initialised with.
    """

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def iter_chunked(self, n: int):
        for chunk in self._chunks:
            yield chunk


class MockResponse:
    """Async context-manager stub for ``aiohttp.ClientResponse``.

    Tracks whether ``__aexit__`` was called (i.e., the context manager was
    properly exited) and whether ``raise_for_status()`` was called.

    Args:
        chunks:     Byte fragments yielded by ``content.iter_chunked()``.
        raise_on_status: If not None, ``raise_for_status()`` raises this.
    """

    def __init__(
        self,
        chunks: list[bytes],
        raise_on_status: Exception | None = None,
    ) -> None:
        self.content = MockContent(chunks)
        self._raise_on_status = raise_on_status
        self.exited = False
        self.raise_for_status_called = False

    def raise_for_status(self) -> None:
        """Synchronous — matches the real aiohttp.ClientResponse API."""
        self.raise_for_status_called = True
        if self._raise_on_status is not None:
            raise self._raise_on_status

    async def __aenter__(self) -> "MockResponse":
        return self

    async def __aexit__(self, *args) -> None:
        self.exited = True


class MockSession:
    """Async context-manager stub for ``aiohttp.ClientSession``.

    Captures the arguments passed to ``post()`` for assertion, returns a
    fixed ``MockResponse`` as a context manager.

    Tracks whether ``__aexit__`` was called.
    """

    def __init__(self, response: MockResponse) -> None:
        self._response = response
        self.exited = False
        # Populated by the first call to post():
        self.post_url: str | None = None
        self.post_kwargs: dict = {}

    def post(self, url: str, **kwargs) -> MockResponse:
        """Capture call args and return the mock response as a context manager."""
        self.post_url = url
        self.post_kwargs = kwargs
        return self._response

    async def __aenter__(self) -> "MockSession":
        return self

    async def __aexit__(self, *args) -> None:
        self.exited = True


def _make_session(
    chunks: list[bytes] | None = None,
    raise_on_status: Exception | None = None,
) -> MockSession:
    """Return a ``MockSession`` pre-wired with the given chunks."""
    return MockSession(MockResponse(chunks or [], raise_on_status=raise_on_status))


def _make_tts(
    monkeypatch,
    voice_id: str = "test-voice",
    model_id: str = "eleven_monolingual_v1",
    output_format: str = "pcm_24000",
) -> tuple[ElevenLabsTTS, MockSession]:
    """Return an (ElevenLabsTTS, MockSession) pair with aiohttp patched."""
    session = _make_session()
    monkeypatch.setattr(
        "agent.tts.elevenlabs.aiohttp.ClientSession",
        lambda: session,
    )
    tts = ElevenLabsTTS(
        api_key="test-api-key",
        voice_id=voice_id,
        model_id=model_id,
        output_format=output_format,
    )
    return tts, session


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInit:
    def test_voice_id_stored(self, monkeypatch):
        """voice_id is accessible as a public attribute."""
        tts, _ = _make_tts(monkeypatch, voice_id="voice-abc")
        assert tts.voice_id == "voice-abc"

    def test_default_model_id(self, monkeypatch):
        """Default model_id is eleven_monolingual_v1."""
        tts, _ = _make_tts(monkeypatch)
        assert tts.model_id == "eleven_monolingual_v1"

    def test_custom_model_id_stored(self, monkeypatch):
        """Custom model_id is stored and used in requests."""
        tts, _ = _make_tts(monkeypatch, model_id="eleven_multilingual_v2")
        assert tts.model_id == "eleven_multilingual_v2"

    def test_default_output_format(self, monkeypatch):
        """Default output_format is pcm_24000."""
        tts, _ = _make_tts(monkeypatch)
        assert tts.output_format == "pcm_24000"

    def test_custom_output_format_stored(self, monkeypatch):
        """Custom output_format is stored and forwarded to the API."""
        tts, _ = _make_tts(monkeypatch, output_format="mp3_44100_128")
        assert tts.output_format == "mp3_44100_128"

    def test_api_key_not_public(self, monkeypatch):
        """api_key is stored privately — not accessible as tts.api_key."""
        tts, _ = _make_tts(monkeypatch)
        assert not hasattr(tts, "api_key"), (
            "api_key must not be stored as a public attribute"
        )


# ---------------------------------------------------------------------------
# synthesize() — audio output
# ---------------------------------------------------------------------------


class TestSynthesize:
    async def test_audio_chunks_yielded_in_order(self, monkeypatch):
        """Non-empty byte chunks are yielded in the order received."""
        session = _make_session(chunks=[b"abc", b"def", b"ghi"])
        monkeypatch.setattr(
            "agent.tts.elevenlabs.aiohttp.ClientSession", lambda: session
        )
        tts = ElevenLabsTTS(api_key="k", voice_id="v")

        result = []
        async for chunk in tts.synthesize("Hello"):
            result.append(chunk)

        assert result == [b"abc", b"def", b"ghi"]

    async def test_empty_chunks_discarded(self, monkeypatch):
        """Empty byte strings are silently discarded and never yielded."""
        session = _make_session(chunks=[b"", b"audio", b"", b"data", b""])
        monkeypatch.setattr(
            "agent.tts.elevenlabs.aiohttp.ClientSession", lambda: session
        )
        tts = ElevenLabsTTS(api_key="k", voice_id="v")

        result = []
        async for chunk in tts.synthesize("Hi"):
            result.append(chunk)

        assert result == [b"audio", b"data"]

    async def test_empty_stream_yields_nothing(self, monkeypatch):
        """An empty response body produces zero chunks without raising."""
        session = _make_session(chunks=[])
        monkeypatch.setattr(
            "agent.tts.elevenlabs.aiohttp.ClientSession", lambda: session
        )
        tts = ElevenLabsTTS(api_key="k", voice_id="v")

        result = []
        async for chunk in tts.synthesize(""):
            result.append(chunk)

        assert result == []

    async def test_synthesize_is_async_generator_not_coroutine(self, monkeypatch):
        """synthesize() returns an async generator — it must NOT be awaited."""
        tts, _ = _make_tts(monkeypatch)

        result = tts.synthesize("Hello")

        assert inspect.isasyncgen(result)
        assert not inspect.iscoroutine(result)

        # Exhaust so the session context manager closes cleanly.
        async for _ in result:
            pass


# ---------------------------------------------------------------------------
# synthesize() — API call parameters
# ---------------------------------------------------------------------------


class TestAPIParameters:
    async def _call(self, monkeypatch, **tts_kwargs) -> MockSession:
        """Helper: create a TTS instance, run synthesize(), return the session."""
        session = _make_session(chunks=[b"x"])
        monkeypatch.setattr(
            "agent.tts.elevenlabs.aiohttp.ClientSession", lambda: session
        )
        tts = ElevenLabsTTS(
            api_key=tts_kwargs.pop("api_key", "test-key"),
            voice_id=tts_kwargs.pop("voice_id", "my-voice"),
            **tts_kwargs,
        )
        async for _ in tts.synthesize("Hello world"):
            pass
        return session

    async def test_url_contains_voice_id(self, monkeypatch):
        """The POST URL encodes the voice_id in the path."""
        session = await self._call(monkeypatch, voice_id="my-voice-id")
        assert "my-voice-id" in session.post_url

    async def test_url_ends_with_stream(self, monkeypatch):
        """The POST URL targets the /stream endpoint."""
        session = await self._call(monkeypatch)
        assert session.post_url.endswith("/stream")

    async def test_api_key_in_header(self, monkeypatch):
        """xi-api-key header carries the api_key value."""
        session = await self._call(monkeypatch, api_key="secret-key-123")
        headers = session.post_kwargs.get("headers", {})
        assert headers.get("xi-api-key") == "secret-key-123"

    async def test_output_format_query_param(self, monkeypatch):
        """output_format is forwarded as a query parameter."""
        session = await self._call(monkeypatch, output_format="pcm_44100")
        params = session.post_kwargs.get("params", {})
        assert params.get("output_format") == "pcm_44100"

    async def test_text_in_payload(self, monkeypatch):
        """The text argument appears in the JSON body."""
        session = _make_session(chunks=[b"x"])
        monkeypatch.setattr(
            "agent.tts.elevenlabs.aiohttp.ClientSession", lambda: session
        )
        tts = ElevenLabsTTS(api_key="k", voice_id="v")
        async for _ in tts.synthesize("Test sentence"):
            pass
        payload = session.post_kwargs.get("json", {})
        assert payload.get("text") == "Test sentence"

    async def test_model_id_in_payload(self, monkeypatch):
        """model_id appears in the JSON body."""
        session = await self._call(monkeypatch, model_id="eleven_multilingual_v2")
        payload = session.post_kwargs.get("json", {})
        assert payload.get("model_id") == "eleven_multilingual_v2"

    async def test_raise_for_status_called(self, monkeypatch):
        """raise_for_status() is invoked before any chunks are yielded."""
        session = _make_session(chunks=[b"audio"])
        monkeypatch.setattr(
            "agent.tts.elevenlabs.aiohttp.ClientSession", lambda: session
        )
        tts = ElevenLabsTTS(api_key="k", voice_id="v")
        async for _ in tts.synthesize("hi"):
            pass
        assert session._response.raise_for_status_called


# ---------------------------------------------------------------------------
# synthesize() — error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    async def test_http_error_propagates(self, monkeypatch):
        """A non-2xx response raises and no chunks are yielded."""
        error = Exception("401 Unauthorized")
        session = _make_session(chunks=[b"should not appear"], raise_on_status=error)
        monkeypatch.setattr(
            "agent.tts.elevenlabs.aiohttp.ClientSession", lambda: session
        )
        tts = ElevenLabsTTS(api_key="bad-key", voice_id="v")

        collected = []
        with pytest.raises(Exception, match="401 Unauthorized"):
            async for chunk in tts.synthesize("hi"):
                collected.append(chunk)

        assert collected == [], "No chunks should be yielded before the error"

    async def test_http_error_before_iteration(self, monkeypatch):
        """raise_for_status() is called before iter_chunked() is entered."""
        error = Exception("422 Unprocessable Entity")
        session = _make_session(raise_on_status=error)
        monkeypatch.setattr(
            "agent.tts.elevenlabs.aiohttp.ClientSession", lambda: session
        )
        tts = ElevenLabsTTS(api_key="k", voice_id="bad-voice-id")

        with pytest.raises(Exception, match="422"):
            async for _ in tts.synthesize("hi"):
                pass

        # raise_for_status was called — confirmed the check happens first
        assert session._response.raise_for_status_called


# ---------------------------------------------------------------------------
# synthesize() — stream cleanup (context manager lifecycle)
# ---------------------------------------------------------------------------


class TestStreamCleanup:
    async def test_session_closed_after_full_iteration(self, monkeypatch):
        """The aiohttp session context manager is exited after all chunks consumed."""
        session = _make_session(chunks=[b"a", b"b"])
        monkeypatch.setattr(
            "agent.tts.elevenlabs.aiohttp.ClientSession", lambda: session
        )
        tts = ElevenLabsTTS(api_key="k", voice_id="v")

        async for _ in tts.synthesize("Hello"):
            pass

        assert session.exited

    async def test_response_closed_after_full_iteration(self, monkeypatch):
        """The response context manager is exited after all chunks consumed."""
        session = _make_session(chunks=[b"a", b"b"])
        monkeypatch.setattr(
            "agent.tts.elevenlabs.aiohttp.ClientSession", lambda: session
        )
        tts = ElevenLabsTTS(api_key="k", voice_id="v")

        async for _ in tts.synthesize("Hello"):
            pass

        assert session._response.exited

    async def test_session_closed_on_early_aclose(self, monkeypatch):
        """Session context manager is exited when the generator is explicitly closed.

        Python's ``async for`` does not call ``aclose()`` automatically on
        ``break`` — the caller must call it explicitly.  This test verifies
        that ``__aexit__`` runs when ``aclose()`` is invoked, which is the
        pattern Issue #8 must follow in the TTS stage.
        """
        session = _make_session(chunks=[b"first", b"second", b"third"])
        monkeypatch.setattr(
            "agent.tts.elevenlabs.aiohttp.ClientSession", lambda: session
        )
        tts = ElevenLabsTTS(api_key="k", voice_id="v")

        gen = tts.synthesize("Hello")
        async for _ in gen:
            break  # bail after first chunk
        await gen.aclose()  # explicit cleanup — triggers context manager __aexit__

        assert session.exited
        assert session._response.exited

    async def test_response_closed_on_exception_during_streaming(self, monkeypatch):
        """Context managers are exited when an exception occurs inside the stream."""

        class BrokenContent:
            async def iter_chunked(self, n: int):
                yield b"first"
                raise ValueError("network error mid-stream")

        broken_response = MockResponse(chunks=[])
        broken_response.content = BrokenContent()

        broken_session = MockSession(broken_response)
        monkeypatch.setattr(
            "agent.tts.elevenlabs.aiohttp.ClientSession", lambda: broken_session
        )
        tts = ElevenLabsTTS(api_key="k", voice_id="v")

        chunks = []
        with pytest.raises(ValueError, match="network error mid-stream"):
            async for chunk in tts.synthesize("Hello"):
                chunks.append(chunk)

        assert chunks == [b"first"]
        assert broken_session.exited
        assert broken_response.exited
