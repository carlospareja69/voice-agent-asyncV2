"""
Tests for agent/pipeline.py

No real hardware, network, or model inference is required.
All external dependencies (sounddevice, STT, LLM, TTS) are replaced with
lightweight fakes.

Mock hierarchy
--------------
FakeSTT     — STTProvider that returns pre-configured results in sequence
FakeLLM     — LLMProvider async generator that yields a fixed token list,
              tracks whether its finally block was executed (aclose_called)
FakeTTS     — TTSProvider async generator that yields a fixed chunk list,
              records each synthesize() call text, tracks aclose_called
_FakeInputStream  — minimal sounddevice.InputStream stand-in for run() tests
_FakeOutputStream — minimal sounddevice.OutputStream stand-in for run() tests

Test coverage
-------------
- Pipeline.__init__: speaker wired to tts_sample_rate, mic wired to audio settings
- _stt_stage: accumulation threshold, empty transcript discard, queue delivery,
              cancellation propagation
- _llm_stage: user/assistant context turns, token streaming, sentinel delivery,
              gen.aclose() on CancelledError, exception resilience
- _tts_stage: sentence assembly on punctuation, flush on sentinel, whitespace guard,
              audio chunks in tts_queue, gen.aclose() on CancelledError,
              multiple sentences per turn
- run(): CancelledError propagates cleanly when the run task is cancelled
"""

import asyncio
from collections.abc import AsyncIterator

import pytest

from agent.llm.base import LLMProvider
from agent.pipeline import Pipeline
from agent.stt.base import STTProvider
from agent.tts.base import TTSProvider
from config.settings import Settings


# ---------------------------------------------------------------------------
# Fake providers
# ---------------------------------------------------------------------------


class FakeSTT(STTProvider):
    """STT provider that returns results from a pre-configured list.

    Returns results in order; returns "" once the list is exhausted.
    Tracks the number of transcribe() calls made.
    """

    def __init__(self, results: list[str]) -> None:
        self._results = list(results)
        self._index = 0
        self.call_count = 0

    async def transcribe(self, audio: bytes) -> str:
        self.call_count += 1
        if self._index < len(self._results):
            result = self._results[self._index]
            self._index += 1
            return result
        return ""


class FakeLLM(LLMProvider):
    """LLM provider that yields a fixed token list.

    ``aclose_called`` is set to True when the generator's finally block runs.
    This happens on normal exhaustion, explicit aclose(), or exception
    propagating through the generator body.
    Tracks all calls via ``generate_calls`` (list of message lists passed).
    """

    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens
        self.aclose_called = False
        self.generate_calls: list[list[dict]] = []

    async def generate(self, messages: list[dict]) -> AsyncIterator[str]:
        self.generate_calls.append(messages)
        try:
            for token in self._tokens:
                yield token
        finally:
            self.aclose_called = True


class FakeTTS(TTSProvider):
    """TTS provider that yields a fixed chunk list for every synthesize() call.

    ``aclose_called`` is set to True when the generator's finally block runs.
    ``synthesize_calls`` records the text argument of each call, in order.
    """

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.aclose_called = False
        self.synthesize_calls: list[str] = []

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        self.synthesize_calls.append(text)
        try:
            for chunk in self._chunks:
                yield chunk
        finally:
            self.aclose_called = True


# ---------------------------------------------------------------------------
# Minimal sounddevice fakes (used only for run() tests)
# ---------------------------------------------------------------------------


class _FakeInputStream:
    def __init__(self, *, callback, **kwargs):
        self.callback = callback

    def __enter__(self) -> "_FakeInputStream":
        return self

    def __exit__(self, *_) -> None:
        pass


class _FakeOutputStream:
    def __init__(self, *, callback, **kwargs):
        self.callback = callback

    def __enter__(self) -> "_FakeOutputStream":
        return self

    def __exit__(self, *_) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(**overrides) -> Settings:
    """Return a minimal valid Settings object with test defaults."""
    defaults = dict(openai_api_key="sk-test", elevenlabs_api_key="el-test")
    defaults.update(overrides)
    return Settings(**defaults)


def _make_pipeline(
    stt: STTProvider | None = None,
    llm: LLMProvider | None = None,
    tts: TTSProvider | None = None,
    **settings_kwargs,
) -> Pipeline:
    """Return a Pipeline wired with fake providers and optional Settings overrides."""
    settings = _make_settings(**settings_kwargs)
    return Pipeline(
        settings=settings,
        stt=stt or FakeSTT([]),
        llm=llm or FakeLLM([]),
        tts=tts or FakeTTS([]),
    )


# ---------------------------------------------------------------------------
# Initialisation tests
# ---------------------------------------------------------------------------


class TestInit:
    def test_speaker_wired_to_tts_sample_rate(self):
        """SpeakerOutput.sample_rate is set from settings.tts_sample_rate."""
        pipeline = _make_pipeline(tts_sample_rate=44100)
        assert pipeline.speaker.sample_rate == 44100

    def test_mic_wired_to_audio_sample_rate(self):
        """MicrophoneInput.sample_rate is set from settings.audio_sample_rate."""
        pipeline = _make_pipeline(audio_sample_rate=8000)
        assert pipeline.mic.sample_rate == 8000

    def test_mic_wired_to_audio_chunk_size(self):
        """MicrophoneInput.chunk_size is set from settings.audio_chunk_size."""
        pipeline = _make_pipeline(audio_chunk_size=512)
        assert pipeline.mic.chunk_size == 512

    def test_token_queue_accepts_none_sentinel(self):
        """token_queue accepts None — the flush sentinel used by the LLM stage."""
        pipeline = _make_pipeline()
        pipeline.token_queue.put_nowait(None)
        assert pipeline.token_queue.get_nowait() is None


# ---------------------------------------------------------------------------
# STT stage tests
# ---------------------------------------------------------------------------


class TestSTTStage:
    async def test_short_audio_not_transcribed(self):
        """A chunk smaller than target_bytes does not trigger transcription."""
        fake_stt = FakeSTT(["hello"])
        # audio_sample_rate=4 → target_bytes = 16
        pipeline = _make_pipeline(stt=fake_stt, audio_sample_rate=4)

        # 8 bytes < 16 — not enough to trigger transcription
        await pipeline.audio_queue.put(b"\x00" * 8)

        task = asyncio.create_task(pipeline._stt_stage())
        await asyncio.sleep(0)  # task processes 8 bytes, blocks at queue.get()

        assert fake_stt.call_count == 0
        assert pipeline.text_queue.empty()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_accumulates_to_target_before_transcribing(self):
        """Transcription is triggered only after target_bytes accumulate."""
        fake_stt = FakeSTT(["hello world"])
        # audio_sample_rate=4 → target_bytes = 16
        pipeline = _make_pipeline(stt=fake_stt, audio_sample_rate=4)

        # Two 8-byte chunks: total = 16 ≥ target — should trigger
        await pipeline.audio_queue.put(b"\x00" * 8)
        await pipeline.audio_queue.put(b"\x00" * 8)

        task = asyncio.create_task(pipeline._stt_stage())
        await asyncio.sleep(0)  # processes both chunks, transcribes, blocks at get()

        assert fake_stt.call_count == 1
        assert pipeline.text_queue.qsize() == 1
        assert pipeline.text_queue.get_nowait() == "hello world"

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_empty_transcript_discarded(self):
        """An empty string returned by STT is not put in text_queue."""
        fake_stt = FakeSTT([""])  # returns "" on first call
        pipeline = _make_pipeline(stt=fake_stt, audio_sample_rate=4)

        await pipeline.audio_queue.put(b"\x00" * 16)  # exactly target_bytes

        task = asyncio.create_task(pipeline._stt_stage())
        await asyncio.sleep(0)

        assert fake_stt.call_count == 1
        assert pipeline.text_queue.empty()  # empty string was discarded

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_non_empty_transcript_reaches_text_queue(self):
        """A non-empty transcript is put in text_queue."""
        fake_stt = FakeSTT(["great day"])
        pipeline = _make_pipeline(stt=fake_stt, audio_sample_rate=4)

        await pipeline.audio_queue.put(b"\x00" * 16)

        task = asyncio.create_task(pipeline._stt_stage())
        await asyncio.sleep(0)

        assert pipeline.text_queue.get_nowait() == "great day"

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_cancelled_error_propagates(self):
        """CancelledError is re-raised so the task group detects shutdown."""
        pipeline = _make_pipeline()

        task = asyncio.create_task(pipeline._stt_stage())
        await asyncio.sleep(0)  # task starts, waits at audio_queue.get()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


# ---------------------------------------------------------------------------
# LLM stage tests
# ---------------------------------------------------------------------------


class TestLLMStage:
    async def test_user_turn_added_to_context(self):
        """The transcript is recorded as a user turn in ContextManager."""
        pipeline = _make_pipeline(llm=FakeLLM(["reply"]))

        await pipeline.text_queue.put("what time is it")
        task = asyncio.create_task(pipeline._llm_stage())
        await asyncio.sleep(0)

        messages = pipeline.context.get_messages()
        assert {"role": "user", "content": "what time is it"} in messages

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_assistant_turn_added_to_context_after_generation(self):
        """The full response is recorded as an assistant turn after generation."""
        pipeline = _make_pipeline(llm=FakeLLM(["Hello", " world"]))

        await pipeline.text_queue.put("hi")
        task = asyncio.create_task(pipeline._llm_stage())
        await asyncio.sleep(0)

        messages = pipeline.context.get_messages()
        assert {"role": "assistant", "content": "Hello world"} in messages

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_tokens_reach_token_queue(self):
        """Each token yielded by generate() is put in token_queue."""
        pipeline = _make_pipeline(llm=FakeLLM(["Hello", " ", "world"]))

        await pipeline.text_queue.put("hi")
        task = asyncio.create_task(pipeline._llm_stage())
        await asyncio.sleep(0)

        tokens = []
        while not pipeline.token_queue.empty():
            item = pipeline.token_queue.get_nowait()
            if item is not None:
                tokens.append(item)

        assert tokens == ["Hello", " ", "world"]

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_none_sentinel_put_after_generation(self):
        """None is the last item in token_queue after a completed generation."""
        pipeline = _make_pipeline(llm=FakeLLM(["A", "B"]))

        await pipeline.text_queue.put("hello")
        task = asyncio.create_task(pipeline._llm_stage())
        await asyncio.sleep(0)

        # Drain everything from token_queue
        items = []
        while not pipeline.token_queue.empty():
            items.append(pipeline.token_queue.get_nowait())

        assert items[-1] is None  # sentinel is last
        assert items[:-1] == ["A", "B"]

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_gen_aclose_called_on_cancelled_error(self):
        """gen.aclose() is called when CancelledError arrives mid-generation.

        The generator is still alive (suspended at yield) when CancelledError
        arrives at await token_queue.put().  Explicit aclose() is required to
        run the generator's finally block and close the HTTP stream promptly.

        A bounded token_queue (maxsize=1) forces the second put() to suspend,
        giving CancelledError a predictable arrival point.
        """
        fake_llm = FakeLLM(["A", "B", "C"])
        pipeline = _make_pipeline(llm=fake_llm)
        pipeline.token_queue = asyncio.Queue(maxsize=1)  # force suspension at 2nd put

        await pipeline.text_queue.put("hello")
        task = asyncio.create_task(pipeline._llm_stage())
        # One tick: "A" put (queue full), task suspended at put("B")
        await asyncio.sleep(0)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert fake_llm.aclose_called

    async def test_cancelled_error_propagates(self):
        """CancelledError re-raises so the pipeline task group detects shutdown."""
        pipeline = _make_pipeline()

        task = asyncio.create_task(pipeline._llm_stage())
        await asyncio.sleep(0)  # task starts, waits at text_queue.get()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_llm_exception_does_not_crash_stage(self):
        """A non-CancelledError from generate() is logged and the stage continues."""

        class ErrorLLM(LLMProvider):
            """LLM that raises immediately — async generator that never yields."""

            async def generate(self, messages: list[dict]) -> AsyncIterator[str]:
                raise ValueError("API error")
                yield  # unreachable — makes this an async generator function

        pipeline = _make_pipeline(llm=ErrorLLM())
        await pipeline.text_queue.put("hello")

        task = asyncio.create_task(pipeline._llm_stage())
        await asyncio.sleep(0)  # processes "hello", generate() raises, continues

        # Stage is still alive — waiting for the next text
        assert not task.done()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_llm_messages_include_context(self):
        """generate() is called with the full context including the system prompt."""
        fake_llm = FakeLLM(["ok"])
        pipeline = _make_pipeline(llm=fake_llm)

        await pipeline.text_queue.put("hello")
        task = asyncio.create_task(pipeline._llm_stage())
        await asyncio.sleep(0)

        assert len(fake_llm.generate_calls) == 1
        messages = fake_llm.generate_calls[0]
        roles = [m["role"] for m in messages]
        assert roles[0] == "system"
        assert "user" in roles

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


# ---------------------------------------------------------------------------
# TTS stage tests
# ---------------------------------------------------------------------------


class TestTTSStage:
    async def test_sentence_assembled_on_punctuation(self):
        """Tokens are accumulated until sentence-ending punctuation triggers flush."""
        fake_tts = FakeTTS([b"audio"])
        pipeline = _make_pipeline(tts=fake_tts)

        await pipeline.token_queue.put("Hello")
        await pipeline.token_queue.put(" world")
        await pipeline.token_queue.put(".")  # triggers flush

        task = asyncio.create_task(pipeline._tts_stage())
        await asyncio.sleep(0)

        assert fake_tts.synthesize_calls == ["Hello world."]

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_exclamation_and_question_mark_trigger_flush(self):
        """! and ? are also sentence-ending characters that trigger flush."""
        fake_tts = FakeTTS([b"audio"])
        pipeline = _make_pipeline(tts=fake_tts)

        await pipeline.token_queue.put("Wow!")
        await pipeline.token_queue.put("Really?")

        task = asyncio.create_task(pipeline._tts_stage())
        await asyncio.sleep(0)

        # Each punctuated token triggers a separate flush
        assert fake_tts.synthesize_calls == ["Wow!", "Really?"]

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_flush_on_sentinel(self):
        """A None sentinel flushes any remaining buffer content."""
        fake_tts = FakeTTS([b"audio"])
        pipeline = _make_pipeline(tts=fake_tts)

        # No punctuation — would never flush without sentinel
        await pipeline.token_queue.put("Hello world")
        await pipeline.token_queue.put(None)  # sentinel

        task = asyncio.create_task(pipeline._tts_stage())
        await asyncio.sleep(0)

        assert fake_tts.synthesize_calls == ["Hello world"]

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_sentinel_on_empty_buffer_does_not_synthesize(self):
        """A None sentinel when the buffer is empty does not call synthesize()."""
        fake_tts = FakeTTS([b"audio"])
        pipeline = _make_pipeline(tts=fake_tts)

        await pipeline.token_queue.put(None)  # sentinel with no preceding tokens

        task = asyncio.create_task(pipeline._tts_stage())
        await asyncio.sleep(0)

        assert fake_tts.synthesize_calls == []  # nothing to flush

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_whitespace_only_buffer_not_synthesized(self):
        """A buffer containing only whitespace is discarded — TTS not called."""
        fake_tts = FakeTTS([b"audio"])
        pipeline = _make_pipeline(tts=fake_tts)

        await pipeline.token_queue.put("   ")  # whitespace
        await pipeline.token_queue.put(None)

        task = asyncio.create_task(pipeline._tts_stage())
        await asyncio.sleep(0)

        assert fake_tts.synthesize_calls == []

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_audio_chunks_reach_tts_queue(self):
        """Byte chunks from synthesize() are put in tts_queue."""
        fake_tts = FakeTTS([b"chunk1", b"chunk2"])
        pipeline = _make_pipeline(tts=fake_tts)

        await pipeline.token_queue.put("Hello.")

        task = asyncio.create_task(pipeline._tts_stage())
        await asyncio.sleep(0)

        items = []
        while not pipeline.tts_queue.empty():
            items.append(pipeline.tts_queue.get_nowait())

        assert items == [b"chunk1", b"chunk2"]

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_multiple_sentences_each_synthesized_separately(self):
        """Two sentence-ending tokens produce two separate synthesize() calls."""
        fake_tts = FakeTTS([b"audio"])
        pipeline = _make_pipeline(tts=fake_tts)

        await pipeline.token_queue.put("Hello.")        # first sentence
        await pipeline.token_queue.put(" How are you?") # second sentence
        await pipeline.token_queue.put(None)            # sentinel (buffer empty)

        task = asyncio.create_task(pipeline._tts_stage())
        await asyncio.sleep(0)

        assert len(fake_tts.synthesize_calls) == 2
        assert fake_tts.synthesize_calls[0] == "Hello."
        assert fake_tts.synthesize_calls[1] == "How are you?"

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_gen_aclose_called_on_cancelled_error(self):
        """gen.aclose() is called when CancelledError arrives mid-synthesis.

        The generator is suspended at yield when CancelledError arrives at
        await tts_queue.put().  Explicit aclose() runs the generator's finally
        block, exiting the aiohttp context managers and releasing the HTTP
        connection.

        A bounded tts_queue (maxsize=1) forces the second put() to suspend.
        """
        fake_tts = FakeTTS([b"chunk1", b"chunk2", b"chunk3"])
        pipeline = _make_pipeline(tts=fake_tts)
        pipeline.tts_queue = asyncio.Queue(maxsize=1)  # force suspension after chunk1

        await pipeline.token_queue.put("Hello.")

        task = asyncio.create_task(pipeline._tts_stage())
        # One tick: "Hello." processed, synthesize() started,
        # chunk1 put (queue full), task suspended at put(chunk2)
        await asyncio.sleep(0)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert fake_tts.aclose_called

    async def test_cancelled_error_propagates(self):
        """CancelledError re-raises so the pipeline task group detects shutdown."""
        pipeline = _make_pipeline()

        task = asyncio.create_task(pipeline._tts_stage())
        await asyncio.sleep(0)  # task starts, waits at token_queue.get()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


# ---------------------------------------------------------------------------
# run() tests
# ---------------------------------------------------------------------------


class TestRun:
    async def test_run_cancels_cleanly(self, monkeypatch):
        """pipeline.run() propagates CancelledError when the run task is cancelled.

        sounddevice is monkeypatched so MicrophoneInput and SpeakerOutput do
        not need real audio hardware.
        """
        monkeypatch.setattr(
            "agent.audio.input.sounddevice.InputStream", _FakeInputStream
        )
        monkeypatch.setattr(
            "agent.audio.output.sounddevice.OutputStream", _FakeOutputStream
        )

        pipeline = _make_pipeline()
        task = asyncio.create_task(pipeline.run())
        await asyncio.sleep(0)  # let all stage tasks start

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
