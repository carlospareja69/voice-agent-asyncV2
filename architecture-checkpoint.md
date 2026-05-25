# Architecture Checkpoint – Mid Sprint Review

## Initial Architectural Diagnosis

A mid-sprint architectural review was performed after completing Issues #1 and #2 following the HITL workflow described in “Running Your AFK Agent”.

The current architecture foundation was considered solid:

* queue-based async pipeline
* provider abstraction pattern
* centralized configuration layer
* modular stage separation

However, several future architectural risks were identified before continuing implementation.

---

## Main Risks Identified

### 1. Incorrect async interface contract

The current `LLMProvider` and `TTSProvider` interfaces use:

```python
async def generate(...) -> AsyncIterator[str]
```

This creates a mismatch between coroutine behavior and async generator behavior, which could produce runtime confusion during streaming implementations.

---

### 2. Missing sentence assembly stage

The current architecture streams individual LLM tokens directly into the TTS stage.

Risk:

* fragmented audio generation
* unnatural speech synthesis
* excessive TTS requests

Recommendation:
Introduce an intermediate sentence assembly stage between LLM and TTS.

---

### 3. Unbounded conversation context growth

`ContextManager` currently stores messages indefinitely.

Risk:

* excessive LLM context size
* future API failures
* degraded runtime performance

Recommendation:
Future implementation of context truncation or sliding window management.

---

## Simulated Architectural Proposals

### Proposal A — Minimal Async Interface Fix

* Replace incorrect async iterator signatures
* Preserve current architecture
* Minimal code changes

Pros:

* low complexity
* fast correction
* minimal disruption

Cons:

* does not improve typing safety

---

### Proposal B — Dual Completion + Streaming Interface

* Separate standard completion from streaming methods
* Allow incremental provider implementations

Pros:

* beginner-friendly abstraction

Cons:

* introduces unnecessary interface complexity
* weak streaming semantics

---

### Proposal C — Typed Messages + Lifecycle Protocol

* Add shared `Message` TypedDict
* Improve type safety across modules
* Introduce provider lifecycle management

Pros:

* stronger architecture contracts
* better long-term maintainability

Cons:

* additional abstraction complexity
* possible premature optimization

---

## Selected Hybrid Solution

A hybrid approach between Proposal A and Proposal C was selected.

Chosen decisions:

* fix async iterator interface contracts
* introduce shared typed message structures
* postpone advanced lifecycle abstractions until real provider implementations exist

Reasoning:
This solution improves correctness and maintainability while avoiding unnecessary complexity during early project stages.

---

## Architectural Lessons

This checkpoint demonstrated the importance of:

* reviewing AI-generated architecture continuously
* detecting future coupling early
* maintaining clean async boundaries
* using HITL supervision before technical debt accumulates

The review process followed the architectural control workflow proposed in “Running Your AFK Agent”.
