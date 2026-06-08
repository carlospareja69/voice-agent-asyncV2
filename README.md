# voice-agent-asyncV2

A minimal real-time conversational voice agent in pure Python using `asyncio`.

**Goal:** Understand the architecture behind frameworks like [Pipecat](https://github.com/pipecat-ai/pipecat) and [LiveKit Agents](https://github.com/livekit/agents) by building a clean, minimal version from scratch.

> This is an educational reference implementation, not a production framework.

---

## Pipeline

```
Microphone → STT → LLM → TTS → Speaker
```

Each stage runs as an independent `asyncio.Task`. Stages communicate only through `asyncio.Queue` — no direct calls between them. This makes every stage independently testable and easy to swap.

```
MicrophoneInput
      ↓  audio_queue  (bytes)
    STT stage
      ↓  text_queue   (str)
    LLM stage
      ↓  token_queue  (str | None)
    TTS stage
      ↓  tts_queue    (bytes)
SpeakerOutput
```

---

## Project Structure

```
voice-agent-asyncV2/
├── agent/
│   ├── pipeline.py          # Orchestrates all stages via asyncio queues
│   ├── audio/
│   │   ├── input.py         # Mic capture → audio_queue
│   │   └── output.py        # tts_queue → speaker playback
│   ├── stt/
│   │   ├── base.py          # STTProvider ABC
│   │   └── whisper.py       # faster-whisper implementation
│   ├── llm/
│   │   ├── base.py          # LLMProvider ABC
│   │   └── openai.py        # OpenAI streaming implementation
│   ├── tts/
│   │   ├── base.py          # TTSProvider ABC
│   │   └── elevenlabs.py    # ElevenLabs streaming implementation
│   └── context/
│       └── manager.py       # Conversation history + system prompt
├── config/
│   └── settings.py          # Env-based configuration
├── tests/
├── main.py                  # Entry point
├── requirements.txt
└── .env.example
```

---

## Prerequisites

- Python 3.11+
- PortAudio (required by `sounddevice`)
  - macOS: `brew install portaudio`
  - Ubuntu/Debian: `sudo apt install portaudio19-dev`
  - Windows: included with the `sounddevice` wheel automatically
- An [OpenAI](https://platform.openai.com) API key
- An [ElevenLabs](https://elevenlabs.io) API key

---

## Quick Start

```bash
git clone https://github.com/carlospareja69/voice-agent-asyncV2.git
cd voice-agent-asyncV2

python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and fill in your API keys

python main.py
```

---

## Configuration

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key |
| `ELEVENLABS_API_KEY` | Yes | — | ElevenLabs API key |
| `WHISPER_MODEL` | No | `base` | Whisper model size (`tiny`, `base`, `small`, …) |
| `LLM_MODEL` | No | `gpt-4o-mini` | OpenAI model name |
| `TTS_VOICE_ID` | No | `21m00Tcm4TlvDq8ikWAM` | ElevenLabs voice ID |
| `TTS_SAMPLE_RATE` | No | `24000` | TTS output sample rate in Hz — must match `ElevenLabsTTS output_format` |

---

## Module Overview

| Module | Responsibility |
|---|---|
| `agent/pipeline.py` | Orchestrates all stages: STT accumulation, LLM streaming with context, inline sentence assembly, TTS synthesis, clean cancellation |
| `agent/audio/input.py` | Captures mic audio, pushes raw PCM bytes to `audio_queue` |
| `agent/audio/output.py` | Reads from `tts_queue`, plays audio through speaker via `sounddevice.OutputStream` |
| `agent/stt/base.py` | `STTProvider` ABC — `async transcribe(audio) -> str` |
| `agent/llm/base.py` | `LLMProvider` ABC — `def generate(messages) -> AsyncIterator[str]` |
| `agent/tts/base.py` | `TTSProvider` ABC — `def synthesize(text) -> AsyncIterator[bytes]` |
| `agent/context/manager.py` | Stores message history, prepends system prompt for each LLM call |
| `config/settings.py` | Loads and validates all env vars at startup |

---

## Audio Format Contracts

Two queue format contracts have been established and must not be changed without updating both producer and consumer in the same commit.

**`audio_queue` — microphone → STT**
- Encoding: `float32`, little-endian
- Sample rate: 16 000 Hz
- Channels: 1 (mono)
- Producer: `MicrophoneInput` / Consumer: `WhisperSTT`
- Reconstruct: `np.frombuffer(chunk, dtype=np.float32)`

**`tts_queue` — TTS → speaker**
- Encoding: signed `int16`, little-endian
- Sample rate: 24 000 Hz
- Channels: 1 (mono)
- Producer: `ElevenLabsTTS` / Consumer: `SpeakerOutput`
- Reconstruct: `np.frombuffer(chunk, dtype=np.int16)`

---

## Streaming Provider Pattern

`LLMProvider.generate()` and `TTSProvider.synthesize()` are **async generators** — calling them returns an `AsyncIterator` directly. Do **not** `await` the call.

```python
# Correct
async for token in llm.generate(messages):
    ...

async for chunk in tts.synthesize(text):
    ...
```

When consuming these in a pipeline stage, hold a reference to the generator and call `await gen.aclose()` before re-raising `CancelledError` to ensure HTTP connections are released promptly:

```python
gen = llm.generate(messages)
try:
    async for token in gen:
        await queue.put(token)
except asyncio.CancelledError:
    await gen.aclose()
    raise
```

---

## Development Workflow

This project follows a **HITL (Human-in-the-Loop)** incremental model:

1. Pick the next GitHub Issue
2. Implement in a feature branch
3. Open a PR referencing the issue
4. Merge after review

---

## Roadmap

- [x] #1 — Initialize project skeleton and dependencies
- [x] #2 — Settings module with env validation
- [x] #3 — Async microphone input (`sounddevice`)
- [x] #4 — Whisper STT provider (`faster-whisper`)
- [x] #5 — OpenAI streaming LLM provider
- [x] #6 — ElevenLabs streaming TTS provider
- [x] #7 — Async speaker output (`sounddevice`)
- [x] #8 — Wire all pipeline stages together
- [ ] #9 — End-to-end smoke test


## AI Development Workflow

This project follows the workflow described in “Running Your AFK Agent”.

Development is performed using Claude Code as the primary implementation assistant in a Human-in-the-Loop (HITL) workflow.

Process followed:

1. Define architecture and project scope
2. Break work into GitHub Issues
3. Implement incrementally issue-by-issue
4. Review AI-generated code manually
5. Maintain architecture quality through continuous human supervision

The goal is not only to build the application, but also to learn how to supervise and coordinate autonomous AI-assisted software development workflows.
