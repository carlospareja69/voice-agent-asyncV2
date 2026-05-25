# Sprint Handoff — Post Issue #3

---

## Completed Issues

| # | Title | Status |
|---|---|---|
| #1 | Initialize project skeleton and dependencies | ✅ Done |
| #2 | Settings module with env validation | ✅ Done |
| #3 | Async microphone input with sounddevice | ✅ Done |

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
| `agent/audio/output.py` | 🔶 Scaffold — `stream()` raises `NotImplementedError` |
| `agent/stt/whisper.py` | 🔶 Scaffold — `transcribe()` raises `NotImplementedError` |
| `agent/llm/openai.py` | 🔶 Scaffold — `generate()` raises `NotImplementedError` |
| `agent/tts/elevenlabs.py` | 🔶 Scaffold — `synthesize()` raises `NotImplementedError` |
| `agent/pipeline.py` | 🔶 Scaffold — all methods raise `NotImplementedError` |
| `main.py` | ✅ Entry point wired — fails at runtime until Issue #8 |
| `tests/test_settings.py` | ✅ 19 tests passing |
| `tests/test_audio_input.py` | ✅ 9 tests passing |

**Total test count: 28 — all passing.**

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

### Threading model (established by Issue #3)
- sounddevice callbacks fire on a PortAudio background thread — not the event loop thread
- The only safe bridge is: `loop.call_soon_threadsafe(queue.put_nowait, data)`
- `loop` must be captured via `asyncio.get_running_loop()` **before** the callback is defined, on the event loop thread
- `indata.tobytes()` must be called inside the callback to create an independent bytes copy before the PortAudio buffer is reused
- This exact pattern will be **mirrored by Issue #7 (SpeakerOutput)**

### Provider pattern
- STT, LLM, and TTS implement ABCs from `base.py` — one abstract method each
- Swapping a provider = one file change, zero pipeline changes

### Cancellation contract
- Every stage coroutine must re-raise `CancelledError` after any local cleanup
- Never swallow `CancelledError` — the pipeline task group depends on seeing it
- Catch `asyncio.CancelledError` specifically, not `Exception` (CancelledError is BaseException in Python 3.8+)

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

## QA Seam Findings (Issue #3 Review)

### Accepted risks (no action required now)

| Risk | Reason accepted |
|---|---|
| Post-cancellation callback fires in the shutdown window | Benign — InputStream `__exit__` is synchronous and stops PortAudio before `asyncio.run()` unwinds. At most one extra chunk in the queue. |
| `logging` call inside audio callback acquires a lock on audio thread | Correct per Python logging docs; only fires on abnormal status events — not on every chunk |

### Deferred to future issues

| Item | Deferred to |
|---|---|
| `QueueFull` exception silently discarded if `queue.maxsize` is ever set | Issue #8 — queue overflow design must be done when `maxsize` is introduced, not independently |
| `PortAudioError` (no input device) path untested | Issue #9 smoke test or integration hardening |
| `sleep(0.05)` in tests is over-conservative; `sleep(0)` would suffice | Cosmetic — safe to leave |

---

## Known Architectural Risks (Pre-Implementation)

These were identified in the mid-sprint architectural review. They are **not yet fixed** and affect Issues #4–#8.

### 🔴 Critical — fix before implementing Issues #5 and #6

**`agent/llm/base.py` and `agent/tts/base.py` have a broken return type.**

```python
# Current — TYPE LIE
async def generate(self, messages: list[dict]) -> AsyncIterator[str]: ...
async def synthesize(self, text: str) -> AsyncIterator[bytes]: ...
```

An `async def` that returns `AsyncIterator` is a coroutine. A coroutine must be `await`ed to get the iterator. But the natural implementation of `generate()` and `synthesize()` is an async generator (with `yield`), which **is itself** the async iterator — callers use `async for token in provider.generate(messages)` directly.  

These two calling conventions are mutually exclusive. The fix is one word: change `async def` to `def`. The concrete implementations remain `async def` with `yield` — they become async generators, which satisfy `AsyncIterator` as their return type.

**Fix this before writing any code in Issues #5 or #6.**

### 🟡 High — design decision required before Issue #8

**Missing sentence assembler stage between LLM and TTS.**

`token_queue` currently carries individual LLM tokens (single words/subwords). TTS providers need sentence-level or phrase-level input to produce natural audio. Feeding individual tokens produces one HTTP request per word with no prosody.

Issue #8 must either:
- Add a sentence assembler task between `token_queue` and the TTS stage, or
- Change `token_queue` to carry full sentences (assembled in the LLM stage before enqueuing)

**Decide the approach when starting Issue #8 — not before.**

### 🟠 Medium — clean up in a single pass

**`ContextManager` has no role validation and unbounded message growth.**

- `add_turn(role, content)` accepts any string as `role` — invalid roles fail later at the OpenAI API with an opaque HTTP 400
- `_messages` grows forever — after ~20 minutes of conversation the context window will overflow
- No `TypedDict` for messages — `list[dict]` is untyped at the API boundary

**Address when implementing Issue #5 (OpenAI LLM), since that's when the context boundary becomes concrete.**

### 🟠 Medium — duplicate defaults maintenance trap

`MicrophoneInput.__init__` has `sample_rate=16000, chunk_size=1024` as defaults.  
`Settings` also has `audio_sample_rate=16000, audio_chunk_size=1024` as defaults.  
`SpeakerOutput.__init__` hardcodes `sample_rate=24000` (not from Settings at all).

These are maintained separately. Fix both audio modules when implementing Issue #7, in a single pass.

---

## Pending Issues

| # | Title | Depends on | Key file |
|---|---|---|---|
| #4 | Whisper STT provider | #3 ✅ (audio format now confirmed) | `agent/stt/whisper.py` |
| #5 | OpenAI streaming LLM provider | — | `agent/llm/openai.py` |
| #6 | ElevenLabs streaming TTS provider | — | `agent/tts/elevenlabs.py` |
| #7 | Async speaker output | — | `agent/audio/output.py` |
| #8 | Wire all pipeline stages | #4–#7 | `agent/pipeline.py` |
| #9 | End-to-end smoke test | #8 | manual + `tests/` |

Issues #4, #5, #6, #7 are **independent** — can be implemented in any order.  
Issue #8 requires all four to be complete.

---

## Intentionally Deferred Technical Debt

| Item | Deferred until |
|---|---|
| Voice Activity Detection (VAD) | After Issue #9 — mic captures continuously; silence hits STT and produces empty strings that propagate to LLM |
| `asyncio.Queue(maxsize=N)` back-pressure | Issue #8 hardening — must be designed together with `QueueFull` error handling in the audio callback |
| `aiohttp.ClientSession` lifecycle in ElevenLabsTTS | Issues #6 (open) + #8 (close on shutdown) |
| `ContextManager` role validation and sliding window | Issue #5 |
| `Message` TypedDict to replace `list[dict]` | Issue #5 — same pass as ContextManager fixes |
| `async def → def` fix on LLM and TTS ABCs | Before Issue #5 or #6 begins (critical) |

---

## Recommended Next Issue

**→ Issue #4: Whisper STT provider** (`agent/stt/whisper.py`)

**Why now:**
- Issue #3 is complete — the audio format is confirmed (`float32`, `chunk_size` frames, mono, bytes)
- Issue #4 was explicitly marked as depending on #3 to know the audio shape before encoding assumptions
- `faster-whisper` is CPU-bound and synchronous — it establishes the `run_in_executor` pattern for blocking inference that no other issue has introduced yet
- Self-contained: no dependency on LLM or TTS

**Key implementation notes for Issue #4:**

```python
# Reconstruct the numpy array from bytes — confirmed format from Issue #3
audio = np.frombuffer(chunk, dtype=np.float32)

# faster-whisper is synchronous — must run in executor to avoid blocking the loop
loop = asyncio.get_running_loop()
segments, _ = await loop.run_in_executor(None, model.transcribe, audio)
transcript = " ".join(seg.text for seg in segments).strip()
```

- Use `asyncio.get_running_loop().run_in_executor(None, ...)` — same loop-capture pattern as Issue #3
- Load the `WhisperModel` once in `__init__`, not per transcription call
- Return empty string `""` for silent/empty transcriptions — the pipeline stage must filter these before sending to LLM
- `device="cpu"` and `compute_type="int8"` are safe defaults; expose them via constructor if needed
- Verify `faster-whisper` wheel availability on Python 3.14 before starting (flagged in handoffs)

---

## Warnings — Future Implementations Must Respect

1. **Fix `async def → def` on `LLMProvider.generate()` and `TTSProvider.synthesize()` before writing any code in Issues #5 or #6.** The current signatures are a calling-convention mismatch that will cause a confusing runtime error.

2. **Every stage task must re-raise `CancelledError`.** `asyncio.gather()` in `pipeline.run()` will not detect shutdown if any stage swallows it.

3. **Blocking inference and I/O must go through `run_in_executor`.** Direct synchronous calls inside `async def` block the entire event loop, freezing all other stages. Whisper inference is the first real test of this constraint.

4. **The `audio_queue` format is fixed.** `float32`, mono, `chunk_size` frames, raw bytes. The STT stage must reconstruct with `np.frombuffer(chunk, dtype=np.float32)`. Changing this requires updating both `input.py` and `whisper.py` in the same PR.

5. **`MicrophoneInput` and `SpeakerOutput` have duplicate default values vs `Settings`.** Always pass `settings.audio_sample_rate` and `settings.audio_chunk_size` explicitly from `pipeline.py`. Do not rely on the constructor defaults.

6. **`ContextManager.add_turn()` does not validate the `role` argument.** Until fixed in Issue #5, only pass `"user"` or `"assistant"` — no other strings.
