# Sprint Handoff — Post Issue #8 (+ pre-Issue #9 consistency cleanup)

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
| #7 | Async speaker output (sounddevice) | ✅ Done |
| #8 | Wire all pipeline stages | ✅ Done |

---

## Current File Status

| File | State |
|---|---|
| `config/settings.py` | ✅ Fully implemented — `tts_sample_rate` added in Issue #7 |
| `agent/context/manager.py` | ✅ Fully implemented |
| `agent/stt/base.py` | ✅ ABC defined |
| `agent/llm/base.py` | ✅ ABC corrected — `def generate` (not `async def`) |
| `agent/tts/base.py` | ✅ ABC corrected — `def synthesize` (not `async def`) |
| `agent/audio/input.py` | ✅ Fully implemented |
| `agent/stt/whisper.py` | ✅ Fully implemented |
| `agent/audio/output.py` | ✅ Fully implemented in Issue #7 |
| `agent/llm/openai.py` | ✅ Fully implemented |
| `agent/tts/elevenlabs.py` | ✅ Fully implemented |
| `agent/pipeline.py` | ✅ Fully implemented in Issue #8 |
| `main.py` | ✅ Entry point wired — `ElevenLabsTTS` receives `output_format` from Settings; `WhisperSTT` receives `sample_rate` from Settings |
| `.env.example` | ✅ `TTS_SAMPLE_RATE` documented |
| `tests/test_settings.py` | ✅ 22 tests passing |
| `tests/test_audio_input.py` | ✅ 9 tests passing |
| `tests/test_audio_output.py` | ✅ 9 tests passing |
| `tests/test_stt_whisper.py` | ✅ 13 tests passing |
| `tests/test_llm_openai.py` | ✅ 13 tests passing |
| `tests/test_tts_elevenlabs.py` | ✅ 23 tests passing |
| `tests/test_context_manager.py` | ✅ 9 tests passing (new in Issue #8) |
| `tests/test_pipeline.py` | ✅ 29 tests passing (new in Issue #8) |

**Total test count: 127 — all passing.**

---

## Consolidated Architecture Decisions

These are settled. Do not change without an explicit architectural discussion.

### Pipeline topology (established by Issue #8)

```
MicrophoneInput
      ↓  audio_queue  (bytes — float32, 16 kHz, mono)
    STT stage         accumulate 1 s windows → WhisperSTT.transcribe()
      ↓  text_queue   (str — transcript)
    LLM stage         ContextManager → OpenAILLM.generate() → tokens + sentinel
      ↓  token_queue  (str | None — token stream, None = end-of-turn flush signal)
    TTS stage         sentence assembler → ElevenLabsTTS.synthesize()
      ↓  tts_queue    (bytes — int16, 24 kHz, mono)
SpeakerOutput
```

- 4 `asyncio.Queue` instances: `audio_queue → text_queue → token_queue → tts_queue`
- 5 `asyncio.Task` instances: `mic`, `stt`, `llm`, `tts`, `speaker`
- Stages **never call each other directly** — only read/write their adjacent queue
- All queues are unbounded (`asyncio.Queue()` with no `maxsize`). Back-pressure is deferred to a future hardening pass.

### `token_queue` sentinel protocol (established by Issue #8)

`token_queue` carries individual LLM tokens followed by `None` after each turn completes. The type is `asyncio.Queue[str | None]`.

- `_llm_stage` puts `None` in two places: after normal generation completes (flush TTS buffer so the final sentence is synthesized), and in the `except Exception` handler if `generate()` fails mid-stream (flush any partial sentence fragment already in the TTS buffer).
- `_tts_stage` treats `None` as: flush whatever is in the sentence buffer, then continue waiting for the next turn. `None` never reaches `ElevenLabsTTS.synthesize()`.
- Whitespace-only sentence fragments are discarded silently (`"".join(buffer).strip()` check).

### Sentence assembly (established by Issue #8)

Sentence assembly is **inline in `_tts_stage`** — no dedicated task or extra queue.

`_SENTENCE_ENDINGS = frozenset(".!?")` — checked against `token[-1]`. The TTS stage accumulates tokens in a `list[str]` buffer and flushes on:
1. A token whose last character is in `_SENTENCE_ENDINGS`
2. A `None` sentinel (end-of-turn)

**Rationale:** Individual tokens (~2–6 chars) cannot be passed to TTS — one HTTP round-trip per token is prohibitive. Assembling at sentence boundaries reduces TTS calls to ~1–3 per response while preserving the streaming benefit: the first sentence starts synthesizing before the LLM has generated the rest.

### Audio accumulation in STT stage (established by Issue #8)

`target_bytes = settings.audio_sample_rate * 4` — 1 second of float32 PCM at the configured sample rate.

- The STT stage uses a `bytearray` accumulator that receives chunks from `audio_queue` until `len(accumulated) >= target_bytes`.
- When the threshold is met, the full buffer is passed to `stt.transcribe()` and the accumulator is cleared.
- Empty transcriptions (silence) are discarded silently — `if transcript:` guard.
- `transcribe()` exceptions are caught, logged, and the audio window is discarded (`continue`) — a single bad chunk does not kill the conversation.

### `gen.aclose()` cancellation contract (established by Issue #5, enforced by Issue #8)

Python's `async for` does **not** call `aclose()` automatically when `CancelledError` arrives in the loop body (at `await queue.put()`). The generator is still alive and holds the HTTP connection open.

Both `_llm_stage` and `_synthesize_sentence` follow this pattern:

```python
gen = provider.generate_or_synthesize(...)
try:
    async for item in gen:
        await queue.put(item)
except asyncio.CancelledError:
    await gen.aclose()   # REQUIRED — releases HTTP stream/aiohttp context managers
    raise
```

`_synthesize_sentence` is a dedicated helper so that `_tts_stage` does not have to track the `synthesize()` generator reference separately.

### `pipeline.run()` shutdown contract (established by Issue #8)

```python
async def run(self) -> None:
    tasks = [asyncio.create_task(...) for each stage]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    finally:
        logger.info("Pipeline stopped.")
```

- `except BaseException` (not `Exception`) is required — `CancelledError` is `BaseException` in Python 3.8+.
- The inner `gather(*tasks, return_exceptions=True)` is intentional: it waits for every task's cleanup (`gen.aclose()`, logging) to complete without propagating secondary cleanup exceptions.
- `run()` re-raises the original exception after cleanup — callers can distinguish normal cancellation from component errors.

### `SpeakerOutput` wiring (completed by Issue #8)

`SpeakerOutput(sample_rate=settings.tts_sample_rate)` — the `Pipeline.__init__` now passes `settings.tts_sample_rate` directly. The duplicate-defaults problem identified in Issue #6 is fully resolved:

- `ElevenLabsTTS(..., output_format=f"pcm_{settings.tts_sample_rate}")` — set in `main.py` (Issue #7)
- `SpeakerOutput(sample_rate=settings.tts_sample_rate)` — set in `Pipeline.__init__` (Issue #8)

Both ends of `tts_queue` now derive their sample rate from the same `Settings` field.

### LLM/TTS ABC calling convention (established by Issue #5)

The `LLMProvider.generate()` and `TTSProvider.synthesize()` abstract methods are declared with `def`, not `async def`. Callers use:

```python
async for token in provider.generate(messages):   # correct — no await
    ...
```

The STT provider remains `async def transcribe()` — it is a coroutine (one await → one value), not a stream. The asymmetry is intentional.

### SpeakerOutput implementation contract (established by Issue #7)

- **Two-layer buffer pattern**: asyncio task → `queue.Queue` (stdlib) → PortAudio callback
- **`bytearray` leftover accumulator** handles chunk/frame size misalignment
- **Silence fill**: `outdata[available_frames:, 0] = 0` — prevents audio glitches when TTS is slow
- PortAudio callback closes over both `_buffer` and `leftover`; lifetime bounded by `with sounddevice.OutputStream(...)` block
- No executor needed: asyncio task just copies references between queues

### `Settings.tts_sample_rate` contract (established by Issue #7)

- `Settings.tts_sample_rate: int = 24000` is the single authoritative source for TTS audio sample rate
- `TTS_SAMPLE_RATE` env var overrides the default
- Both `ElevenLabsTTS` and `SpeakerOutput` derive their sample rate from this field

### ElevenLabsTTS implementation contract (established by Issue #6)

- Session strategy: **session-per-request** (`async with aiohttp.ClientSession()` inside `synthesize()`). No `close()` method. `Pipeline.run()` does not call `tts.close()` at shutdown.
- `api_key` is `self._api_key` (private) — not a public attribute
- `response.raise_for_status()` is synchronous in aiohttp — do not `await` it
- `voice_settings` (`stability=0.5`, `similarity_boost=0.75`) are hardcoded constants

### `tts_queue` audio format contract (established by Issue #6)

```
Encoding    : signed 16-bit integer (int16), little-endian
Sample rate : 24 000 Hz (configurable via TTS_SAMPLE_RATE)
Channels    : 1 (mono)
```

Reconstruct with: `np.frombuffer(chunk, dtype=np.int16)`

**Changing `output_format` on `ElevenLabsTTS` invalidates this contract.** Both the constructor call and `SpeakerOutput` configuration must be updated in the same commit.

### Audio format contract (established by Issue #3)

```
Encoding    : float32, little-endian
Sample rate : 16 000 Hz
Channels    : 1 (mono)
```

Reconstruct with: `np.frombuffer(chunk, dtype=np.float32)`

### Threading model (established by Issue #3, confirmed by Issue #4)

- sounddevice callbacks fire on PortAudio's thread — input bridge via `loop.call_soon_threadsafe()`; output bridge via `queue.Queue`
- CPU-bound synchronous work (Whisper inference) runs via `loop.run_in_executor(None, fn)`
- `loop` must be captured via `asyncio.get_running_loop()` before entering any background thread context
- Lazy generators from synchronous libraries must be fully consumed inside the executor callable

### Cancellation contract

- Every stage coroutine catches `asyncio.CancelledError`, logs, performs cleanup (`gen.aclose()`), and re-raises
- Never swallow `CancelledError` — `pipeline.run()` depends on seeing it
- Catch `asyncio.CancelledError` specifically, not `Exception`

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

### Issue #8 — Accepted risks

| Risk | Reason accepted |
|---|---|
| No back-pressure: all queues are unbounded (`asyncio.Queue()` with no `maxsize`) | Deferred to a future hardening pass. Back-pressure requires co-designed `maxsize` + `QueueFull` error handling across all stages. For a sequential voice agent with one active conversation, unbounded queues are safe in practice. |
| `token_queue` sentinel (`None`) does not carry error context — TTS stage cannot distinguish "LLM finished normally" from "LLM failed mid-stream" | Not needed for pipeline correctness: both cases produce a flush, which is the correct behavior. If error telemetry is needed later, replace the sentinel with a typed dataclass. |
| Sentence assembly uses only `.`, `!`, `?` — no Unicode sentence boundaries | Sufficient for English conversational text. Extend `_SENTENCE_ENDINGS` if multilingual or technical text (ellipsis, etc.) is needed. |
| STT stage discards the full 1-second window on any `transcribe()` exception — partial audio lost | Correct trade-off: partial audio produces worse transcription than silence. A future VAD-aware accumulator could recover the window. |
| No VAD: silence hits WhisperSTT, producing empty transcriptions that are discarded | Existing behavior. VAD is intentionally deferred to post-Issue #9. Empty-string guard (`if transcript:`) prevents silent propagation. |
| `ContextManager` has no role validation and unbounded message growth | Pipeline only calls `add_turn("user", ...)` and `add_turn("assistant", ...)`. Role validation deferred to a future hardening pass. Context window overflow is a real concern for long conversations. |

### Issue #7 — Accepted risks (still applicable)

| Risk | Reason accepted |
|---|---|
| PortAudio callback fires on shutdown before `OutputStream.__exit__` runs | Benign — at most one extra audio glitch during teardown. |
| `queue.Queue` (`_buffer`) can grow unbounded if asyncio task produces faster than PortAudio consumes | Cannot happen in practice: PortAudio runs in real time and ElevenLabs is network-bound. |
| No validation that `TTS_SAMPLE_RATE` is a known ElevenLabs value | Intentionally deferred. Add `VALID_TTS_SAMPLE_RATES` allowlist when a non-24000 value is needed. |

### Issue #6 — Accepted risks (still applicable)

| Risk | Reason accepted |
|---|---|
| `voice_settings` hardcoded in request payload | Promote to constructor parameters when voice-quality tuning is needed. |
| Session-per-request discards TCP connection pooling | Acceptable for sequential single-conversation pipeline. |
| `model_id="eleven_monolingual_v1"` may produce suboptimal results for non-English text | Pass `model_id="eleven_multilingual_v2"` at construction for multilingual use. |

### Issue #5 — Accepted risks (still applicable)

| Risk | Reason accepted |
|---|---|
| `stream.close()` raising in `finally` replaces in-flight exception | SDK documents `close()` as idempotent. Fix with `contextlib.suppress(Exception)` if observed. |
| Multiple concurrent `generate()` calls share `self.client` — connection pool exhaustion untested | Not a concern for single-conversation agent. |
| `CancelledError` live-cancellation path in `generate()` not covered by isolated unit test | Covered by the end-to-end smoke test in Issue #9. |

### Issue #4 — Accepted risks (still applicable)

| Risk | Reason accepted |
|---|---|
| `np.frombuffer` returns read-only array passed to faster-whisper | Works today. Add `.copy()` if a `ValueError: assignment destination is read-only` surfaces. |

---

## Known Architectural Risks

### ✅ Resolved — LLM/TTS ABC signature (fixed in Issue #5)

Both abstract methods corrected from `async def` to `def`.

### ✅ Resolved — LLM stage cancellation cleanup (implemented in Issue #8)

`_llm_stage` calls `await gen.aclose()` before re-raising `CancelledError`. `_synthesize_sentence` does the same for the TTS generator. Both HTTP connections are released promptly on cancellation.

### ✅ Resolved — Sentence assembler between LLM and TTS (implemented in Issue #8)

Inline sentence assembly in `_tts_stage`. Flushes on `_SENTENCE_ENDINGS` or `None` sentinel.

### ✅ Resolved — `SpeakerOutput` not wired through `Settings.tts_sample_rate` (fixed in Issue #8)

`Pipeline.__init__` now passes `sample_rate=settings.tts_sample_rate` to `SpeakerOutput`. Full resolution of the duplicate-defaults problem.

### 🟠 Medium — `ContextManager` role validation and unbounded growth

- `add_turn(role, content)` accepts any string as `role`
- `_messages` grows forever — context window overflow after extended conversations
- No `TypedDict` for messages — `list[dict]` is untyped at the API boundary

Deferred past Issue #9. The pipeline only passes `"user"` or `"assistant"` so correctness is not at risk. Address in a hardening pass.

### 🟡 Low — `WhisperSTT.sample_rate` unused field

`__init__` accepts and stores `sample_rate` (now correctly wired from `settings.audio_sample_rate` in `main.py`), but `_sync_transcribe` never passes it to `model.transcribe()`. faster-whisper always assumes 16 kHz input. A caller passing `sample_rate=8000` gets no error, just degraded quality. Address in a future hardening pass (Option A: warning; B: remove; C: doc-only).

---

## Pending Issues

| # | Title | Depends on | Key file |
|---|---|---|---|
| #9 | End-to-end smoke test | #8 ✅ | `main.py` + manual verification |

All component and integration issues are complete. Issue #9 is now unblocked.

---

## Intentionally Deferred Technical Debt

| Item | Deferred until |
|---|---|
| Voice Activity Detection (VAD) | After Issue #9 — silence hits STT producing empty strings discarded by `if transcript:` |
| `asyncio.Queue(maxsize=N)` back-pressure | Future hardening — must be designed together with `QueueFull` error handling |
| `ContextManager` role validation and sliding window | Post-Issue #9 hardening |
| `Message` TypedDict to replace `list[dict]` | Post-Issue #9 hardening |
| `WhisperSTT.sample_rate` unused field | Post-Issue #9 hardening (Option A/B/C) |
| `beam_size` hardcoded in `_sync_transcribe` | Future performance-tuning issue |
| `np.frombuffer` read-only array in Whisper | Add `.copy()` if `ValueError` surfaces |
| `CancelledError` unit test for `generate()` live-cancellation | Issue #9 end-to-end smoke test |
| `stream.close()` exception masking in `finally` | Fix only if observed |
| `voice_settings` hardcoded in `ElevenLabsTTS` | Promote when voice-quality tuning is needed |
| Unicode sentence boundary detection | Extend `_SENTENCE_ENDINGS` when multilingual use is needed |
| `TTS_SAMPLE_RATE` allowlist validation | Add when a non-24000 value is observed in practice |

---

## Recommended Next Issue

**→ Issue #9: End-to-end smoke test** — all pipeline stages are wired and individually tested.

**Key focus for Issue #9:**

1. **Manual smoke test** — run `python main.py`, speak into the microphone, verify: mic → transcription → LLM response → audio playback. Check for audio glitches, latency, and error handling.

2. **Automated smoke test options:**
   - Full pipeline test with real providers using environment keys (skip in CI if keys unavailable)
   - Integration test with fake providers that exercises the complete `run()` → cancel sequence end-to-end (useful even in CI)

3. **Live-cancellation path** — verify `gen.aclose()` is called correctly under real cancellation conditions (not just in isolated unit tests with bounded queues).

4. **First-word latency** — verify sentence-boundary flushing produces audio before the LLM finishes generating the full response.

5. **Silence handling** — confirm empty transcripts do not trigger spurious LLM calls.

---

## Warnings — Future Implementations Must Respect

1. **Every stage task must re-raise `CancelledError`.** `asyncio.gather()` in `pipeline.run()` will not detect shutdown if any stage swallows it.

2. **Blocking inference and I/O must go through `run_in_executor`.** Direct synchronous calls inside `async def` block the entire event loop.

3. **Lazy generators from synchronous libraries must be consumed inside the executor callable.** Never return a generator to the event loop for iteration.

4. **The `audio_queue` format is fixed.** `float32`, mono, `chunk_size` frames, raw bytes. Changing this requires updating both `input.py` and `whisper.py` in the same commit.

5. **The `tts_queue` format is fixed.** `int16`, mono, 24 kHz. Changing `output_format` on `ElevenLabsTTS` requires updating `SpeakerOutput` in the same commit.

6. **`ContextManager.add_turn()` does not validate the `role` argument.** Only pass `"user"` or `"assistant"`.

7. **`WhisperSTT.sample_rate` is stored but unused.** Do not pass values other than 16000.

8. **Both `_llm_stage` and `_synthesize_sentence` call `await gen.aclose()` before re-raising `CancelledError`.** This pattern must be preserved in any future modification to these methods.

9. **`token_queue` carries `str | None`.** `None` is the end-of-turn sentinel. Any stage or test that reads from `token_queue` must handle `None`.

10. **`ElevenLabsTTS` has no `close()` method.** Do not call `tts.close()` at shutdown.
