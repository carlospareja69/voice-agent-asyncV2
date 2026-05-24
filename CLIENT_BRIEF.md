# Client Brief – Voice Agent Async

## Project Overview

Voice Agent Async is an educational real-time conversational voice agent built with pure Python and asyncio.

The goal of the project is to deeply understand the architecture behind modern real-time voice AI frameworks such as Pipecat and LiveKit Agents by implementing a minimal modular system from scratch.

The system follows the pipeline:

Microphone → Speech-to-Text → LLM → Text-to-Speech → Speaker

## Problem Statement

Many developers use high-level AI frameworks without understanding how real-time voice pipelines work internally. This project aims to explore and learn the architecture, async orchestration, and modular design patterns behind conversational voice agents.

## Proposed Solution

Build a modular async-first architecture where each stage of the pipeline operates independently using asyncio queues and decoupled provider interfaces.

The system is intentionally designed incrementally through GitHub Issues using a Human-in-the-Loop (HITL) workflow with Claude Code as the primary implementation assistant.

## Technical Goals

* Learn asyncio-based orchestration
* Understand real-time pipeline architectures
* Explore modular provider abstractions
* Practice AI-assisted software development workflows
* Maintain a scalable and testable architecture

## Technologies

* Python
* asyncio
* OpenAI API
* Whisper
* ElevenLabs
* GitHub Issues
* Claude Code

## Development Workflow

This project follows the workflow described in “Running Your AFK Agent”, where development is performed incrementally through:

* issue-driven development
* human supervision
* iterative architecture reviews
* AI-assisted implementation
