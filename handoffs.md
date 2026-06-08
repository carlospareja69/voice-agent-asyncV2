# Sprint Handoff — Post Issue #4

---

## Completed Issues

| # | Title | Status |
|---|---|---|
| #1 | Initialize project skeleton and dependencies | ✅ Done |
| #2 | Settings module with env validation | ✅ Done |
| #3 | Async microphone input with sounddevice | ✅ Done |
| #4 | Whisper STT provider | ✅ Done |

---

## Current File Status

| File | State |
|---|---|
| `config/settings.py` | ✅ Fully implemented |
| `agent/context/manager.py` | ✅ Fully implemented |
| `agent/stt/base.py` | ✅ ABC defined |
| `agent/llm/base.py` | ✅ ABC defined — ⚠️ broken signature (see Warnings) |
| `agent/tts/base.py` | ✅ ABC defined — ⚠️ broken signature (see Warnings) |
| `agent/audio/input.py` | ✅ Fully implemented |
| `agent/stt/whisper.py` | ✅ Fully implemented |
| `agent/audio/output.py` | 🔶 Scaffold — `stream()` raises `NotImplementedError` |
| `agent/llm/openai.py` | 🔶 Scaffold — `generate()` raises `NotImplementedError` |
| `agent/tts/elevenlabs.py` | 🔶 Scaffold — `synthesize()` raises `NotImplementedError` |
| `agent/pipeline.py` | 🔶 Scaffold — all methods raise `NotImplementedError` |
| `main.py` | ✅ Entry point wired — fails at runtime until Issue #8 |
| `tests/test_settings.py` | ✅ 19 tests passing |
| `tests/test_audio_input.py` | ✅ 9 tests passing |
| `tests/test_stt_whisper.py` | ✅ 13 tests passing |

**Total test count: 41 — all passing.**

---

## Consolidated Architecture Decisions

These are settled. Do not change without an explicit architectural discussion.

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

## Known Architectural Risks (Pre-Implementation)

These were identified in the mid-sprint architectural review. They affect Issues #5–#8.

### 🔴 Critical — fix before starting Issues #5 or #6

**`agent/llm/base.py` and `agent/tts/base.py` have a broken return type.**

```python
# Current — TYPE LIE
async def generate(self, messages: list[dict]) -> AsyncIterator[str]: ...
async def synthesize(self, text: str) -> AsyncIterator[bytes]: ...
```

An `async def` that returns `AsyncIterator` is a coroutine. The natural implementation is an async generator (with `yield`), which **is itself** the async iterator — callers use `async for token in provider.generate(messages)` directly, no `await`. These two calling conventions are mutually exclusive.

Fix: change `async def` to `def` on both abstract methods. One word per file.

**This must happen before writing any code in Issues #5 or #6.**

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
| #5 | OpenAI streaming LLM provider | ABC fix (🔴 Critical above) | `agent/llm/openai.py` |
| #6 | ElevenLabs streaming TTS provider | ABC fix (🔴 Critical above) | `agent/tts/elevenlabs.py` |
| #7 | Async speaker output | — | `agent/audio/output.py` |
| #8 | Wire all pipeline stages | #5, #6, #7 | `agent/pipeline.py` |
| #9 | End-to-end smoke test | #8 | manual + `tests/` |

Issues #5, #6, #7 are independent of each other. Issues #5 and #6 both require the ABC fix first.  
Issue #8 requires all three to be complete.

---

## Intentionally Deferred Technical Debt

| Item | Deferred until |
|---|---|
| Voice Activity Detection (VAD) | After Issue #9 — silence hits STT producing empty strings that propagate to LLM |
| `asyncio.Queue(maxsize=N)` back-pressure | Issue #8 hardening — must be designed together with `QueueFull` error handling |
| `aiohttp.ClientSession` lifecycle in ElevenLabsTTS | Issues #6 (open) + #8 (close on shutdown) |
| `ContextManager` role validation and sliding window | Issue #5 |
| `Message` TypedDict to replace `list[dict]` | Issue #5 — same pass as ContextManager fixes |
| `async def → def` fix on LLM and TTS ABCs | Prerequisite before Issue #5 or #6 begins |
| `WhisperSTT.sample_rate` unused field | Before Issue #8 — see options under Known Risks above |
| `beam_size` hardcoded in `_sync_transcribe` | Future performance-tuning issue |
| `np.frombuffer` read-only array in Whisper | Add `.copy()` if faster-whisper write errors surface |
| Chunk accumulation before STT | Issue #8 — pipeline stage must collect enough audio before calling `transcribe()` |

---

## Recommended Next Issue

**→ Fix LLM and TTS ABC signatures (prerequisite micro-task), then Issue #7.**

**The ABC fix first:**
Two one-line changes — `async def generate` → `def generate` and `async def synthesize` → `def synthesize`. These are blocking prerequisites for Issues #5 and #6. Do them as a single small commit before any other issue starts.

**Then Issue #7: Async speaker output** (`agent/audio/output.py`)

**Why #7 before #5 or #6:**
- No ABC fix dependency — `SpeakerOutput` has no ABC
- Mirrors Issue #3 (microphone input) exactly: callback-based sounddevice, `call_soon_threadsafe`, `CancelledError` re-raise, executor-free
- Establishes the full audio path (input ✅ → STT ✅ → ... → speaker 🔶) from the edges inward
- After #7, Issues #5 and #6 can be implemented in either order with the complete audio frame in place

**Key implementation notes for Issue #7:**

- `SpeakerOutput` is the inverse of `MicrophoneInput`: reads from a queue, writes to the device
- Use `sounddevice.OutputStream` with a callback — callback runs on PortAudio's thread
- The callback reads from the queue; use a synchronous bridge (a `queue.Queue`, not `asyncio.Queue`) internally, or read directly in a loop with `asyncio.Queue.get_nowait()` inside the callback
  - ⚠️ **Important design choice**: sounddevice output callbacks are time-critical (must return before the next buffer deadline). Using `asyncio.Queue` across a thread boundary here is the inverse of Issue #3 — the audio thread needs to *pull* from the queue, not push to it. A `queue.Queue` (threading-safe) as an internal buffer, fed by the asyncio task, is the cleaner pattern
- Fix `sample_rate=24000` hardcoding — read from `Settings.audio_sample_rate` or add a dedicated `Settings.tts_sample_rate` field
- Re-raise `CancelledError` after closing the stream

---

## Warnings — Future Implementations Must Respect

1. **Fix `async def → def` on `LLMProvider.generate()` and `TTSProvider.synthesize()` before writing any code in Issues #5 or #6.** The current signatures are a calling-convention mismatch.

2. **Every stage task must re-raise `CancelledError`.** `asyncio.gather()` in `pipeline.run()` will not detect shutdown if any stage swallows it.

3. **Blocking inference and I/O must go through `run_in_executor`.** Direct synchronous calls inside `async def` block the entire event loop. Whisper demonstrates this correctly — it is the reference pattern.

4. **Lazy generators from synchronous libraries must be consumed inside the executor callable.** Never return a generator to the event loop for iteration. `_sync_transcribe` demonstrates this correctly.

5. **The `audio_queue` format is fixed.** `float32`, mono, `chunk_size` frames, raw bytes. STT stage reconstructs with `np.frombuffer(chunk, dtype=np.float32)`. Changing this requires updating both `input.py` and `whisper.py` in the same commit.

6. **`ContextManager.add_turn()` does not validate the `role` argument.** Until fixed in Issue #5, only pass `"user"` or `"assistant"`.

7. **`WhisperSTT.sample_rate` is stored but unused.** Do not pass values other than 16000 — they produce no error but degrade transcription quality silently.

8. **Chunk accumulation is a pipeline responsibility (Issue #8).** `WhisperSTT.transcribe()` accepts any-length bytes, but 64 ms chunks (1024 frames at 16 kHz) produce poor results. The STT stage loop must accumulate enough audio before calling `transcribe()`.
