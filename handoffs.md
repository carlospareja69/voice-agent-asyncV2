# Sprint Handoff — Post Issue #6

---

## Completed Issues

| # | Title | Status |
|---|---|---|
| #1 | Initialize project skeleton and dependencies | ✅ Done |
| #2 | Settings module with env validation | ✅ Done |
| #3 | Async microphone input with sounddevice | ✅ Done |
| #4 | Whisper STT provider | ✅ Done |
| #5 | OpenAI streaming LLM provider | ✅ Done |
| #6 | ElevenLabs streaming TTS provider | ✅ Done |

---

## Current File Status

| File | State |
|---|---|
| `config/settings.py` | ✅ Fully implemented |
| `agent/context/manager.py` | ✅ Fully implemented |
| `agent/stt/base.py` | ✅ ABC defined |
| `agent/llm/base.py` | ✅ ABC corrected — `def generate` (not `async def`) |
| `agent/tts/base.py` | ✅ ABC corrected — `def synthesize` (not `async def`) |
| `agent/audio/input.py` | ✅ Fully implemented |
| `agent/stt/whisper.py` | ✅ Fully implemented |
| `agent/audio/output.py` | 🔶 Scaffold — `stream()` raises `NotImplementedError` |
| `agent/llm/openai.py` | ✅ Fully implemented |
| `agent/tts/elevenlabs.py` | ✅ Fully implemented |
| `agent/pipeline.py` | 🔶 Scaffold — all methods raise `NotImplementedError` |
| `main.py` | ✅ Entry point wired — fails at runtime until Issue #8 |
| `tests/test_settings.py` | ✅ 19 tests passing |
| `tests/test_audio_input.py` | ✅ 9 tests passing |
| `tests/test_stt_whisper.py` | ✅ 13 tests passing |
| `tests/test_llm_openai.py` | ✅ 13 tests passing |
| `tests/test_tts_elevenlabs.py` | ✅ 23 tests passing |

**Total test count: 77 — all passing.**

---

## Consolidated Architecture Decisions

These are settled. Do not change without an explicit architectural discussion.

### LLM/TTS ABC calling convention (established by Issue #5)

The `LLMProvider.generate()` and `TTSProvider.synthesize()` abstract methods are declared with `def`, not `async def`. This is the correct signature for a method whose concrete implementation is an async generator function. Callers use:

```python
async for token in provider.generate(messages):   # correct — no await
    ...
```

Declaring `async def` on these methods was a type lie: the annotation claimed `AsyncIterator` but the calling convention would have required `await`. The fix makes the contract unambiguous. The STT provider remains `async def transcribe()` because it is a coroutine (one await → one value), not a stream. The asymmetry is intentional — it reflects the difference between "await a result" and "iterate a stream."

**Consequence for all future providers:** Any class implementing `LLMProvider` or `TTSProvider` must define its streaming method as an async generator function (containing `yield`). Returning a synchronous iterator will pass ABC registration but fail at runtime.

### OpenAILLM implementation contract (established by Issue #5)

- `AsyncOpenAI` client is created once in `__init__` and reused across calls — connection pooling is preserved.
- `api_key` is NOT stored on `self`; it lives only inside the SDK client. This prevents accidental key exposure via repr or logging.
- `generate()` opens a streaming request (`stream=True`), yields non-None delta content tokens, and always closes the HTTP stream in a `try/finally` block.
- Chunks with empty `choices` lists (keep-alive frames) and `None` delta content (role-metadata frames) are silently skipped — both are normal SSE protocol frames.
- `stream = await client.chat.completions.create(...)` is placed **outside** the `try` block deliberately. There is no suspension point between the `await` returning and `try:` being entered, so `CancelledError` cannot arrive in that gap. The `finally` runs if and only if `stream` was successfully assigned.

### ElevenLabsTTS implementation contract (established by Issue #6)

- Session strategy: **Option C — session-per-request.** A new `aiohttp.ClientSession` is created inside each `synthesize()` call using `async with`. No persistent session state. No `close()` method on `ElevenLabsTTS`. **Issue #8 does NOT need to call `tts.close()` at shutdown.** This supersedes the pre-implementation recommendation in the earlier handoff.
- `api_key` is stored as `self._api_key` (private). It is passed in the `xi-api-key` request header and never assigned to a public attribute. There is no `tts.api_key` — this is intentional.
- `output_format` defaults to `"pcm_24000"`. Changing it changes the audio format of every byte yielded. The `SpeakerOutput` configuration in Issue #7 must match.
- `voice_settings` (`stability=0.5`, `similarity_boost=0.75`) are hardcoded constants. Not constructor parameters. Accepted technical debt — promote when needed.
- `response.raise_for_status()` is **synchronous** in aiohttp. It must not be `await`-ed. It is called immediately after the response is received, before any iteration begins.
- Empty byte fragments from `iter_chunked()` are discarded silently. Empty chunks are a normal aiohttp delivery artefact on chunked transfer encoding boundaries.

### `tts_queue` audio format contract (established by Issue #6)

This is the authoritative description of items stored in `tts_queue`. It is the direct parallel of the `audio_queue` format contract established in Issue #3.

```
Encoding    : signed 16-bit integer (int16), little-endian
Sample rate : 24 000 Hz
Channels    : 1 (mono)
```

Every `bytes` object yielded by `ElevenLabsTTS.synthesize()` (with default `output_format="pcm_24000"`) is a fragment of this stream. Downstream consumers reconstruct with:

```python
np.frombuffer(chunk, dtype=np.int16)
```

**Changing `output_format` on `ElevenLabsTTS` invalidates this contract.** Both the constructor call and the `SpeakerOutput` configuration must be updated together in the same commit. Issue #7 must configure its `sounddevice.OutputStream` with `dtype="int16"`, `samplerate=24000`, `channels=1`.

### Async generator finalization contract (established by Issue #5)

Python's `async for` does **not** call `aclose()` on an async generator when a `break` or consumer exception exits the loop body. Finalization is scheduled by the event loop's async-gen finalizer hook (`sys.set_asyncgen_finalizer`, registered by `asyncio.run()`) and runs asynchronously — at the next GC cycle or reliably at loop shutdown.

The `try/finally` block inside `generate()` only runs immediately when:
1. The stream is exhausted normally (all chunks consumed), **or**
2. `CancelledError` is thrown into the generator while it is suspended at `yield`, **or**
3. An exception propagates from within the stream iteration itself, **or**
4. The caller explicitly calls `await gen.aclose()`

**Issue #8 must call `await gen.aclose()` in the LLM stage's cancellation handler.** See the Critical requirement below.

### Pipeline topology
- 4 `asyncio.Queue` instances: `audio_queue → text_queue → token_queue → tts_queue`
- Each stage is an independent `asyncio.Task`
- Stages **never call each other directly** — only read/write their adjacent queue
- All queues are typed: `Queue[bytes]`, `Queue[str]`

### Audio format contract (established by Issue #3)
- dtype: `float32`, channels: `1` (mono)
- Each queue item is a raw bytes object: `chunk_size` float32 samples
- STT stage reconstructs with: `np.frombuffer(chunk, dtype=np.float32)`
- This is the single authoritative description of `audio_queue` items

### Threading model (established by Issue #3, confirmed by Issue #4)
- sounddevice callbacks fire on PortAudio's thread — bridge via `loop.call_soon_threadsafe()`
- CPU-bound synchronous work (Whisper inference) runs via `loop.run_in_executor(None, fn)`
- `loop` must be captured via `asyncio.get_running_loop()` before entering any background thread context
- Lazy generators returned by synchronous libraries (e.g., `model.transcribe()`) must be **fully consumed inside the executor callable** — never returned to the event loop for iteration

### Provider pattern
- STT, LLM, and TTS implement ABCs from `base.py` — one abstract method each
- Swapping a provider = one file change, zero pipeline changes

### Cancellation contract
- Every stage coroutine must re-raise `CancelledError` after any local cleanup
- Never swallow `CancelledError` — the pipeline task group depends on seeing it
- Catch `asyncio.CancelledError` specifically, not `Exception` (it is `BaseException` in Python 3.8+)

### Model loading contract (established by Issue #4)
- Expensive synchronous initialization (model weight loading) belongs in `__init__`
- `__init__` runs in `build_pipeline()`, which is called before `asyncio.run()`
- The event loop must never pay model loading costs

### Configuration contract
- `Settings.from_env()` is the only valid production constructor
- All config values are validated before `asyncio.run()` is called
- Stages receive config via constructor arguments, not by importing Settings directly

### Startup sequence (fixed)
```
Settings.from_env()         # sync — validates env before event loop starts
build_pipeline(settings)    # sync — constructs all providers and queues
asyncio.run(pipeline.run()) # async — event loop starts here
```

---

## QA Seam Findings

### Issue #3 — Accepted risks

| Risk | Reason accepted |
|---|---|
| Post-cancellation callback fires in the shutdown window | Benign — InputStream `__exit__` is synchronous and stops PortAudio before `asyncio.run()` unwinds. At most one extra chunk in the queue. |
| `logging` call inside audio callback acquires a lock on audio thread | Correct per Python logging docs; only fires on abnormal status events — not on every chunk |

### Issue #6 — Accepted risks

| Risk | Reason accepted |
|---|---|
| `voice_settings` (`stability`, `similarity_boost`) hardcoded in the request payload | Configurable in the ElevenLabs API but not needed for this project. Promote to constructor parameters when voice-quality tuning becomes a requirement. |
| Session-per-request discards TCP connection pooling | Acceptable for a sequential voice-agent pipeline making one TTS call at a time. If concurrent TTS calls were ever added, replace with a persistent shared session. |
| `output_format` mismatch between `ElevenLabsTTS` and `SpeakerOutput` is not caught at startup | Both are plain constructor parameters with no cross-validation. If they diverge, the speaker plays noise or crashes. Mitigation: wire both through `Settings.tts_sample_rate` (deferred to Issue #7). |
| `model_id="eleven_monolingual_v1"` may produce suboptimal results for non-English text | Known limitation of the default model. Pass `model_id="eleven_multilingual_v2"` at construction for multilingual use. |

### Issue #5 — Accepted risks

| Risk | Reason accepted |
|---|---|
| `stream.close()` raising in the `finally` block replaces the in-flight exception (including `CancelledError`) | SDK documents `close()` as idempotent; network errors on close are low-probability. If it surfaces, the fix is `contextlib.suppress(Exception)` around the close call. |
| Multiple concurrent `generate()` calls share `self.client` — connection pool exhaustion untested | `AsyncOpenAI` is designed for concurrent use. Not a concern for the current single-conversation agent; defer to future hardening. |
| ABC cannot enforce at registration time that implementations are async generators | Python limitation. A `def generate()` returning a sync iterator passes ABC registration but fails at runtime. Acceptable — any misimplementation fails loudly on first use. |
| `CancelledError` path through `generate()` is not covered by an isolated unit test | The `try/finally` correctness is verified by `test_stream_closed_on_exception_during_streaming`. The live-cancellation path will be covered by the end-to-end test in Issue #9. |

### Issue #4 — Accepted risks

| Risk | Reason accepted |
|---|---|
| `np.frombuffer` returns a read-only array passed to faster-whisper | Works today; faster-whisper reads but never writes the input array. If a future version writes to it, the crash is a clear `ValueError: assignment destination is read-only`. Low probability; fix with `.copy()` if it surfaces. |
| `model.transcribe()` exception types not documented or tested | Correct to let exceptions propagate. Issue #8 must decide whether the STT stage catches and continues or shuts down. |

### Deferred to future issues (both issues)

| Item | Deferred to |
|---|---|
| `QueueFull` exception silently discarded if `queue.maxsize` is set | Issue #8 — must be designed together with maxsize introduction |
| `PortAudioError` (no input device) path untested | Issue #9 smoke test or hardening |
| `beam_size=5` hardcoded, not configurable | Future performance-tuning issue |

---

## Known Architectural Risks

These were identified in the mid-sprint architectural review and updated after Issue #5 QA.

### ✅ Resolved — LLM/TTS ABC signature (fixed in Issue #5)

`agent/llm/base.py` and `agent/tts/base.py` have been corrected. Both abstract methods now use `def` instead of `async def`. The calling convention is unambiguous. See the Architecture Decisions section above for full rationale.

### 🔴 Critical for Issue #8 — LLM stage cancellation cleanup

**The `_llm_stage` task in `agent/pipeline.py` must explicitly call `await gen.aclose()` before re-raising `CancelledError`.**

Python does not automatically finalize an async generator when `async for` is abandoned via `break` or exception propagation from the loop body. Without explicit `aclose()`, the `try/finally` block inside `generate()` does not run promptly — the HTTP stream to OpenAI may remain open until the event loop shuts down.

The required pattern:

```python
async def _llm_stage(self) -> None:
    while True:
        text = await self.text_queue.get()
        self.context.add_turn("user", text)
        gen = self.llm.generate(self.context.get_messages())
        try:
            async for token in gen:
                await self.token_queue.put(token)
        except asyncio.CancelledError:
            await gen.aclose()   # ← REQUIRED — closes the HTTP stream promptly
            raise
```

**This is a hard requirement. Omitting it is not a latent bug for a single-conversation agent (the event loop finalizer catches it at shutdown), but it becomes a connection pool leak in any service that restarts the pipeline without restarting the process.**

The same requirement applies to `_tts_stage` and `ElevenLabsTTS.synthesize()`:

```python
async def _tts_stage(self) -> None:
    while True:
        sentence = await self.sentence_queue.get()  # assembled sentence (Issue #8)
        gen = self.tts.synthesize(sentence)
        try:
            async for audio_chunk in gen:
                await self.tts_queue.put(audio_chunk)
        except asyncio.CancelledError:
            await gen.aclose()   # ← REQUIRED — exits the aiohttp context managers
            raise
```

### 🟡 High — design decision required before Issue #8

**Missing sentence assembler stage between LLM and TTS.**

`token_queue` carries individual LLM tokens. TTS providers need sentence-level input for natural audio. Issue #8 must either add a sentence assembler task or change the LLM stage to enqueue full sentences.

### 🟡 High — address before Issue #8

**`WhisperSTT.sample_rate` is stored but never used.**

`__init__` accepts `sample_rate` and stores it, but `_sync_transcribe` never passes it to `model.transcribe()`. faster-whisper always assumes 16 kHz input. A caller passing `sample_rate=8000` gets no error, just degraded transcription quality. This creates a false constructor contract.

Fix before Issue #8 (where the STT stage loop is written and the call is embedded):
- Option A: add `if self.sample_rate != 16000: logger.warning(...)` to signal misuse
- Option B: remove the `sample_rate` parameter entirely (simplest)
- Option C: document explicitly in the docstring that the value is informational only

### 🟠 Medium — clean up in a single pass

**`ContextManager` has no role validation and unbounded message growth.**

- `add_turn(role, content)` accepts any string as `role`
- `_messages` grows forever — context window overflow after extended conversations
- No `TypedDict` for messages — `list[dict]` is untyped at the API boundary

**Address when implementing Issue #5 (OpenAI LLM).**

### 🟠 Medium — duplicate defaults maintenance trap

`MicrophoneInput.__init__` has `sample_rate=16000, chunk_size=1024` as defaults.
`Settings` also has `audio_sample_rate=16000, audio_chunk_size=1024` as defaults.
`SpeakerOutput.__init__` hardcodes `sample_rate=24000` (not from Settings at all).

Fix both audio modules in a single pass when implementing Issue #7.

---

## Pending Issues

| # | Title | Depends on | Key file |
|---|---|---|---|
| #7 | Async speaker output | — | `agent/audio/output.py` |
| #8 | Wire all pipeline stages | #5 ✅, #6 ✅, #7 | `agent/pipeline.py` |
| #9 | End-to-end smoke test | #8 | manual + `tests/` |

Issue #7 is the only remaining blocker before Issue #8.

---

## Intentionally Deferred Technical Debt

| Item | Deferred until |
|---|---|
| Voice Activity Detection (VAD) | After Issue #9 — silence hits STT producing empty strings that propagate to LLM |
| `asyncio.Queue(maxsize=N)` back-pressure | Issue #8 hardening — must be designed together with `QueueFull` error handling |
| ~~`aiohttp.ClientSession` lifecycle in ElevenLabsTTS~~ | ✅ Resolved in Issue #6 — session-per-request, no close() needed |
| `ContextManager` role validation and sliding window | Issue #8 |
| `Message` TypedDict to replace `list[dict]` | Issue #8 — same pass as ContextManager fixes |
| ~~`async def → def` fix on LLM and TTS ABCs~~ | ✅ Done in Issue #5 |
| `WhisperSTT.sample_rate` unused field | Before Issue #8 — see options under Known Risks above |
| `beam_size` hardcoded in `_sync_transcribe` | Future performance-tuning issue |
| `np.frombuffer` read-only array in Whisper | Add `.copy()` if faster-whisper write errors surface |
| Chunk accumulation before STT | Issue #8 — pipeline stage must collect enough audio before calling `transcribe()` |
| `CancelledError` unit test for `generate()` | Issue #9 — live-cancellation path covered by end-to-end smoke test |
| `stream.close()` exception masking in `finally` | Fix only if observed — add `contextlib.suppress(Exception)` around close call |
| `voice_settings` hardcoded in `ElevenLabsTTS` | Promote to constructor parameters when voice-quality tuning is needed |
| `output_format` / `SpeakerOutput.sample_rate` not cross-validated at startup | Wire both through `Settings.tts_sample_rate` in Issue #7 |

---

## Recommended Next Issue

**→ Issue #7: Async speaker output** (`agent/audio/output.py`) — the only remaining blocker before Issue #8.

Issue #7 completes the full audio frame: input ✅ → STT ✅ → LLM ✅ → TTS ✅ → speaker 🔶.

**Key implementation notes for Issue #7:**

**Audio format input:**
`SpeakerOutput` reads from `tts_queue`, which carries raw int16 PCM bytes at 24 kHz mono (established by Issue #6). Configure `sounddevice.OutputStream` with:
- `dtype="int16"`
- `samplerate=24000`
- `channels=1`

Fix `sample_rate=24000` hardcoding in the scaffold — add `Settings.tts_sample_rate` (or rename the existing `audio_sample_rate` convention). Both `ElevenLabsTTS(output_format="pcm_24000")` and `SpeakerOutput(sample_rate=24000)` must be wired through the same `Settings` field. This resolves the duplicate-defaults maintenance trap flagged in the pre-#5 architectural review.

**Threading model — the critical inversion from Issue #3:**
`MicrophoneInput` pushed bytes onto an `asyncio.Queue` from a PortAudio callback thread (using `call_soon_threadsafe`). `SpeakerOutput` is the inverse: it must *pull* bytes from the queue and write them to the device. The PortAudio output callback runs on a real-time audio thread under a hard deadline — it cannot `await` anything.

**Required pattern**: use a `queue.Queue` (stdlib, thread-safe) as an internal buffer between the asyncio event loop and the PortAudio callback:
1. The asyncio task (`stream()` coroutine) reads from `tts_queue` (`asyncio.Queue[bytes]`) and puts items into a `queue.Queue[bytes]` (thread-safe)
2. The PortAudio output callback reads from the `queue.Queue` using `get_nowait()` — no blocking, no asyncio
3. If the `queue.Queue` is empty, the callback writes silence (zeros) to avoid audio glitches

**Cancellation:**
Same pattern as Issue #3 — catch `CancelledError` inside `stream()`, close the `OutputStream`, re-raise.

**No executor needed:**
Unlike Issue #4 (Whisper), sounddevice output callbacks are not CPU-bound. The asyncio task just drains the `tts_queue` and feeds the internal buffer. No `run_in_executor` required.

**Issue #8 notes (for reference when both #7 is done):**
- `ElevenLabsTTS` has no `close()` method — Issue #8 does not call `tts.close()`
- TTS stage must call `await gen.aclose()` before re-raising `CancelledError` (same rule as LLM stage)
- Sentence assembler between `token_queue` and TTS must be designed in Issue #8 — `synthesize()` takes full sentences, not individual tokens

---

## Warnings — Future Implementations Must Respect

1. ~~Fix `async def → def` on `LLMProvider.generate()` and `TTSProvider.synthesize()`~~ — **Done in Issue #5.**

2. **Every stage task must re-raise `CancelledError`.** `asyncio.gather()` in `pipeline.run()` will not detect shutdown if any stage swallows it.

3. **Blocking inference and I/O must go through `run_in_executor`.** Direct synchronous calls inside `async def` block the entire event loop. Whisper demonstrates this correctly — it is the reference pattern.

4. **Lazy generators from synchronous libraries must be consumed inside the executor callable.** Never return a generator to the event loop for iteration. `_sync_transcribe` demonstrates this correctly.

5. **The `audio_queue` format is fixed.** `float32`, mono, `chunk_size` frames, raw bytes. STT stage reconstructs with `np.frombuffer(chunk, dtype=np.float32)`. Changing this requires updating both `input.py` and `whisper.py` in the same commit.

6. **`ContextManager.add_turn()` does not validate the `role` argument.** Until fixed in Issue #8, only pass `"user"` or `"assistant"`.

7. **`WhisperSTT.sample_rate` is stored but unused.** Do not pass values other than 16000 — they produce no error but degrade transcription quality silently.

8. **Chunk accumulation is a pipeline responsibility (Issue #8).** `WhisperSTT.transcribe()` accepts any-length bytes, but 64 ms chunks (1024 frames at 16 kHz) produce poor results. The STT stage loop must accumulate enough audio before calling `transcribe()`.

9. **Both `_llm_stage` and `_tts_stage` in Issue #8 must call `await gen.aclose()` before re-raising `CancelledError`.** This applies to every async generator in the pipeline. The context managers inside `synthesize()` and the `try/finally` inside `generate()` do not run automatically when `async for` is abandoned — both require explicit `aclose()` from their stage.
