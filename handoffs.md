# Sprint Handoff

## Completed Issues

| # | Title | Status |
|---|---|---|
| #1 | Initialize project skeleton and dependencies | ✅ Done |
| #2 | Settings module with env validation | ✅ Done |

---

## Current File Status

| File | State |
|---|---|
| `config/settings.py` | ✅ Fully implemented |
| `tests/test_settings.py` | ✅ 19 tests, all passing |
| `agent/context/manager.py` | ✅ Fully implemented |
| `agent/stt/base.py` | ✅ ABC defined |
| `agent/llm/base.py` | ✅ ABC defined |
| `agent/tts/base.py` | ✅ ABC defined |
| `agent/pipeline.py` | 🔶 Scaffold only — all methods raise `NotImplementedError` |
| `agent/audio/input.py` | 🔶 Scaffold only — `stream()` raises `NotImplementedError` |
| `agent/audio/output.py` | 🔶 Scaffold only — `stream()` raises `NotImplementedError` |
| `agent/stt/whisper.py` | 🔶 Scaffold only — `transcribe()` raises `NotImplementedError` |
| `agent/llm/openai.py` | 🔶 Scaffold only — `generate()` raises `NotImplementedError` |
| `agent/tts/elevenlabs.py` | 🔶 Scaffold only — `synthesize()` raises `NotImplementedError` |
| `main.py` | ✅ Entry point wired — will fail at runtime until Issue #8 is done |

---

## Established Architecture Decisions

These decisions are **settled** — do not change them without a deliberate architectural discussion.

**Pipeline topology**
- 4 `asyncio.Queue` instances connect 5 stages: `audio_queue → text_queue → token_queue → tts_queue`
- Stages are independent `asyncio.Task` objects; they **never call each other directly**
- All queues are typed: `Queue[bytes]`, `Queue[str]`

**Provider pattern**
- STT, LLM, and TTS each have an ABC in `base.py` with a single async method
- Concrete implementations live in sibling files (`whisper.py`, `openai.py`, `elevenlabs.py`)
- Swapping a provider = one file change, zero pipeline changes

**Blocking I/O rule**
- CPU-bound or synchronous I/O (Whisper inference, `sounddevice` callbacks) must run in `loop.run_in_executor()`
- Nothing that blocks the OS thread goes into a bare `async def`

**Config contract**
- `Settings.from_env()` is the only valid constructor in production code
- By the time `asyncio.run()` is called, all env vars are validated and all types are correct
- API keys carry `repr=False` — they must never appear in logs or tracebacks

**Startup sequence**
```
Settings.from_env()         # sync — before event loop
build_pipeline(settings)    # sync — constructs providers and queues
asyncio.run(pipeline.run()) # async — event loop starts here
```

---

## Pending Issues

| # | Title | Depends on | Key file |
|---|---|---|---|
| #3 | Async microphone input | — | `agent/audio/input.py` |
| #4 | Whisper STT provider | #3 (audio chunk shape) | `agent/stt/whisper.py` |
| #5 | OpenAI streaming LLM provider | — | `agent/llm/openai.py` |
| #6 | ElevenLabs streaming TTS provider | — | `agent/tts/elevenlabs.py` |
| #7 | Async speaker output | — | `agent/audio/output.py` |
| #8 | Wire all pipeline stages | #3–#7 | `agent/pipeline.py` |
| #9 | End-to-end smoke test | #8 | manual + `tests/` |

Issues #3, #5, #6, #7 are **independent** and can be implemented in parallel.  
Issue #4 should follow #3 so the audio chunk shape (dtype, layout) is confirmed first.  
Issue #8 is the integration step — requires all of #3–#7 to be complete.

---

## Constraints Future Issues Must Preserve

1. **No cross-stage direct calls.** Every inter-stage communication goes through a queue. If a stage needs data from another stage, add a queue — do not add a method call.

2. **Blocking I/O in executors only.** `sounddevice` streams and `faster-whisper` inference are synchronous. Wrap in `asyncio.get_event_loop().run_in_executor(None, ...)`. Never call them bare inside `async def`.

3. **`Settings` is the single source of truth for config.** Do not hardcode sample rates, model names, or voice IDs inside stage implementations. Read from `settings.*` passed through constructors.

4. **ABCs define the interface contract.** Each new provider must subclass `STTProvider`, `LLMProvider`, or `TTSProvider`. Do not add public methods to concrete classes that callers depend on — keep the interface narrow and swappable.

5. **`ContextManager` owns conversation history.** No stage should hold its own message list. The pipeline passes `context` to the LLM stage; nothing else touches it.

6. **Graceful shutdown via `asyncio.CancelledError`.** When `pipeline.run()` is implemented in Issue #8, cancellation must propagate cleanly to all tasks. Do not swallow `CancelledError`.

---

## Known Technical Debt / Future Risks

| Item | Risk | Notes |
|---|---|---|
| No VAD (Voice Activity Detection) | Medium | `MicrophoneInput` captures continuously. Without silence detection, STT will transcribe silence and empty strings will hit the LLM. Add a VAD stub or threshold in Issue #3; address properly later. |
| Token-level `token_queue` vs. TTS sentence chunks | Medium | TTS providers perform better on sentence-level input, not individual tokens. Issue #8 will need a sentence assembler between `token_queue` and the TTS stage, or the queue type changes. Decide at Issue #8. |
| Unbounded queues | Low (now) | All `asyncio.Queue` instances have no `maxsize`. If TTS lags LLM generation, `tts_queue` grows without limit. Add `maxsize=` in a future hardening issue. |
| `aiohttp.ClientSession` lifecycle | Low (now) | `ElevenLabsTTS` must open a session and close it on shutdown. Issue #6 creates it; Issue #8 must wire the teardown into `pipeline.run()`. |
| Python 3.14 in use | Low | Confirmed working on 3.14.3. `faster-whisper` binary wheels may lag new Python releases — verify wheel availability during Issue #4. |

---

## Recommended Next Issue

**→ Issue #3: Async microphone input** (`agent/audio/input.py`)

**Why first:**
- Entirely self-contained, no dependencies on any unimplemented module
- Establishes the `sounddevice` + executor pattern that Issue #7 (speaker output) will mirror directly
- Confirms the audio dtype and chunk layout before Issue #4 (Whisper STT) encodes assumptions about them

**Key implementation notes:**
- Use `sounddevice.InputStream` with a callback
- The callback is **synchronous** — use `queue.put_nowait()` from inside it (thread-safe for `asyncio.Queue`)
- Audio dtype should be `float32` — matches faster-whisper's expected input format
- `chunk_size` and `sample_rate` are already wired into `MicrophoneInput.__init__` from `Settings`
- Add a test that mocks `sounddevice` and verifies chunks land in the queue
