"""Voice agent pipeline — orchestrates all stages via asyncio queues.

Stage topology
--------------
    MicrophoneInput
         ↓ audio_queue   (bytes — float32, 16 kHz, mono)
       STT stage          accumulate 1 s windows → WhisperSTT.transcribe()
         ↓ text_queue    (str — transcript)
       LLM stage          ContextManager → OpenAILLM.generate() → tokens + sentinel
         ↓ token_queue   (str | None — token stream, None = end-of-turn flush signal)
       TTS stage          sentence assembler → ElevenLabsTTS.synthesize()
         ↓ tts_queue     (bytes — int16, 24 kHz, mono)
    SpeakerOutput

Queue protocols
---------------
- audio_queue  : bytes produced by MicrophoneInput at chunk_size granularity
- text_queue   : non-empty transcript strings only (empty strings are discarded)
- token_queue  : individual LLM tokens followed by None sentinel after each turn
- tts_queue    : raw PCM audio chunks from ElevenLabs, any size

Cancellation contract
---------------------
Every stage catches asyncio.CancelledError, performs local cleanup (logging,
gen.aclose() where applicable), and re-raises.  pipeline.run() cancels all
tasks on any exception or on its own cancellation and waits for cleanup to
complete before returning.
"""

import asyncio
import logging

from agent.audio.input import MicrophoneInput
from agent.audio.output import SpeakerOutput
from agent.context.manager import ContextManager
from agent.llm.base import LLMProvider
from agent.stt.base import STTProvider
from agent.tts.base import TTSProvider
from config.settings import Settings

logger = logging.getLogger(__name__)

# Characters that mark the end of a spoken sentence.  When a token's last
# character is one of these, the TTS stage flushes its buffer immediately so
# synthesis starts before the LLM has finished generating the full response.
_SENTENCE_ENDINGS = frozenset(".!?")


class Pipeline:
    """Orchestrates the full voice agent pipeline.

    Each stage runs as an independent asyncio.Task.  Stages communicate only
    through asyncio.Queue — no stage calls another directly.  This keeps every
    stage independently testable and easy to swap.

    Args:
        settings: Runtime configuration (sample rates, model names, API keys).
        stt:      STT provider — must implement STTProvider.transcribe().
        llm:      LLM provider — must implement LLMProvider.generate().
        tts:      TTS provider — must implement TTSProvider.synthesize().
    """

    def __init__(
        self,
        settings: Settings,
        stt: STTProvider,
        llm: LLMProvider,
        tts: TTSProvider,
    ) -> None:
        self.settings = settings
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.context = ContextManager()

        # Queues — all unbounded.  Back-pressure design is deferred to a
        # future hardening pass (requires co-designed maxsize + QueueFull
        # error handling across all stages).
        self.audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.text_queue: asyncio.Queue[str] = asyncio.Queue()
        self.token_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self.tts_queue: asyncio.Queue[bytes] = asyncio.Queue()

        # Audio I/O — wired through Settings so both ends share the same
        # sample-rate source.
        self.mic = MicrophoneInput(
            sample_rate=settings.audio_sample_rate,
            chunk_size=settings.audio_chunk_size,
        )
        self.speaker = SpeakerOutput(sample_rate=settings.tts_sample_rate)

    # ------------------------------------------------------------------
    # STT stage
    # ------------------------------------------------------------------

    async def _stt_stage(self) -> None:
        """Accumulate audio chunks and transcribe to text.

        Reads raw float32 PCM bytes from audio_queue.  Accumulates chunks
        until 1 second of audio is available, then transcribes.  Empty
        transcriptions (silence, short audio) are discarded silently.
        Transcription errors are logged and the window is discarded so a
        single bad chunk does not kill the conversation.

        Audio accumulation target:
            audio_sample_rate frames × 4 bytes/frame (float32) = 1 second
        """
        accumulated = bytearray()
        # 1 second of float32 audio at the configured sample rate.
        target_bytes = self.settings.audio_sample_rate * 4

        try:
            while True:
                chunk = await self.audio_queue.get()
                accumulated.extend(chunk)

                if len(accumulated) < target_bytes:
                    continue

                audio_bytes = bytes(accumulated)
                accumulated.clear()

                try:
                    transcript = await self.stt.transcribe(audio_bytes)
                except Exception:
                    logger.exception(
                        "STT transcription failed — discarding audio window."
                    )
                    continue

                if transcript:
                    logger.info("Transcript: %r", transcript)
                    await self.text_queue.put(transcript)
                else:
                    logger.debug(
                        "Empty transcription — discarding (silence or short audio)."
                    )

        except asyncio.CancelledError:
            logger.info("STT stage stopped.")
            raise

    # ------------------------------------------------------------------
    # LLM stage
    # ------------------------------------------------------------------

    async def _llm_stage(self) -> None:
        """Generate a streaming LLM response for each transcript.

        For each transcript received from text_queue:
        1. Adds a user turn to the conversation context.
        2. Streams tokens from generate() into token_queue.
        3. Puts None (flush sentinel) after generation completes so the TTS
           stage synthesises any remaining sentence fragment.
        4. Adds the full assistant response to context (only on normal
           completion — partial responses from cancelled or failed calls are
           not stored).

        Generator cleanup
        -----------------
        gen is held as a local variable so that aclose() can be called
        explicitly on CancelledError.  Python's async-for does not call
        aclose() automatically when CancelledError arrives in the loop body
        (i.e. at await token_queue.put()).  Without explicit aclose(), the
        try/finally inside generate() does not run and the HTTP stream to
        OpenAI stays open until event loop shutdown.
        """
        try:
            while True:
                text = await self.text_queue.get()
                self.context.add_turn("user", text)

                gen = self.llm.generate(self.context.get_messages())
                response_tokens: list[str] = []

                try:
                    async for token in gen:
                        response_tokens.append(token)
                        await self.token_queue.put(token)

                except asyncio.CancelledError:
                    # CancelledError arrived while the consumer was suspended
                    # at await token_queue.put() — the generator is still
                    # alive and needs explicit finalisation.
                    await gen.aclose()
                    raise

                except Exception:
                    logger.exception(
                        "LLM generation failed — flushing TTS buffer and continuing."
                    )
                    # Send the sentinel so the TTS stage flushes any partial
                    # sentence fragment already in its buffer.
                    await self.token_queue.put(None)
                    continue

                # Normal completion: flush TTS buffer and update context.
                await self.token_queue.put(None)
                if response_tokens:
                    self.context.add_turn("assistant", "".join(response_tokens))

        except asyncio.CancelledError:
            logger.info("LLM stage stopped.")
            raise

    # ------------------------------------------------------------------
    # TTS stage (includes sentence assembler)
    # ------------------------------------------------------------------

    async def _tts_stage(self) -> None:
        """Assemble LLM tokens into sentences and synthesise audio.

        Reads tokens from token_queue.  Accumulates tokens into a sentence
        buffer.  The buffer is flushed to ElevenLabsTTS.synthesize() when:
        - A token's last character is sentence-ending punctuation (. ! ?)
        - A None sentinel is received (end of LLM turn — flush remainder)

        Whitespace-only fragments are discarded silently.

        Sentence assembly rationale
        ---------------------------
        token_queue carries individual LLM sub-word tokens (~2–6 chars each).
        Passing individual tokens to TTS would incur one HTTP round-trip per
        token (prohibitive latency) and produce unnatural prosody (each token
        synthesised in isolation lacks pitch and rhythm context).  Assembling
        at sentence boundaries reduces TTS calls to ~1–3 per response while
        preserving the streaming benefit: the first sentence is synthesised
        before the LLM has generated the rest of the response.
        """
        buffer: list[str] = []

        try:
            while True:
                token = await self.token_queue.get()

                if token is not None:
                    buffer.append(token)
                    # Flush on sentence-ending punctuation.  Checking the last
                    # character handles both "Hello." (token includes punct)
                    # and "." (punct as its own token).
                    needs_flush = bool(token) and token[-1] in _SENTENCE_ENDINGS
                else:
                    # None sentinel: flush whatever is left in the buffer so
                    # the final sentence fragment of the response is synthesised.
                    needs_flush = True

                if needs_flush and buffer:
                    sentence = "".join(buffer).strip()
                    buffer.clear()
                    if sentence:
                        await self._synthesize_sentence(sentence)

        except asyncio.CancelledError:
            logger.info("TTS stage stopped.")
            raise

    async def _synthesize_sentence(self, sentence: str) -> None:
        """Synthesise a single sentence and put audio chunks in tts_queue.

        Holds a reference to the synthesize() generator so that aclose() can
        be called explicitly on CancelledError — ensuring the aiohttp session
        and HTTP connection are released promptly rather than waiting for event
        loop shutdown.

        Args:
            sentence: Complete sentence or phrase to synthesise.  Must be
                      non-empty and stripped of leading/trailing whitespace.
        """
        logger.debug("Synthesizing: %r", sentence)
        gen = self.tts.synthesize(sentence)
        try:
            async for chunk in gen:
                await self.tts_queue.put(chunk)
        except asyncio.CancelledError:
            await gen.aclose()
            raise

    # ------------------------------------------------------------------
    # Pipeline runner
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Start all pipeline stages as concurrent tasks and run until done.

        Creates one asyncio.Task per stage, runs them concurrently via
        asyncio.gather(), and handles shutdown:

        - If any stage raises an unexpected exception, gather() re-raises it
          here, all remaining tasks are cancelled, and we wait for their
          cleanup to finish before re-raising.

        - If this coroutine is cancelled (e.g. KeyboardInterrupt → Ctrl+C),
          the same cancel-and-wait sequence runs.  Each stage's
          except asyncio.CancelledError block (including gen.aclose() calls)
          runs to completion before run() returns.

        The inner gather(*tasks, return_exceptions=True) is intentional: it
        waits for every task's cleanup code to run without propagating any
        secondary exceptions from the cleanup itself.
        """
        logger.info("Pipeline starting.")
        tasks = [
            asyncio.create_task(self.mic.stream(self.audio_queue),    name="mic"),
            asyncio.create_task(self._stt_stage(),                    name="stt"),
            asyncio.create_task(self._llm_stage(),                    name="llm"),
            asyncio.create_task(self._tts_stage(),                    name="tts"),
            asyncio.create_task(self.speaker.stream(self.tts_queue),  name="speaker"),
        ]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            # Wait for all tasks to finish their cancellation cleanup.
            # return_exceptions=True prevents secondary cleanup exceptions
            # from hiding the original error.
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            logger.info("Pipeline stopped.")
