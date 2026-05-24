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


class Pipeline:
    """
    Orchestrates the full voice agent pipeline.

    Stages are connected via asyncio queues — no stage calls another directly:

        MicrophoneInput
             ↓ audio_queue (bytes)
           STT stage
             ↓ text_queue (str)
           LLM stage
             ↓ token_queue (str)
           TTS stage
             ↓ tts_queue (bytes)
        SpeakerOutput
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

        self.audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.text_queue: asyncio.Queue[str] = asyncio.Queue()
        self.token_queue: asyncio.Queue[str] = asyncio.Queue()
        self.tts_queue: asyncio.Queue[bytes] = asyncio.Queue()

        self.mic = MicrophoneInput(
            sample_rate=settings.audio_sample_rate,
            chunk_size=settings.audio_chunk_size,
        )
        self.speaker = SpeakerOutput()

    async def _stt_stage(self) -> None:
        """Read raw audio from audio_queue, transcribe, push text to text_queue."""
        # TODO: Issue #8 — implement transcription loop
        raise NotImplementedError

    async def _llm_stage(self) -> None:
        """Read transcript from text_queue, generate response, push tokens to token_queue."""
        # TODO: Issue #8 — implement LLM generation loop with context management
        raise NotImplementedError

    async def _tts_stage(self) -> None:
        """Read tokens from token_queue, synthesize audio, push bytes to tts_queue."""
        # TODO: Issue #8 — implement TTS synthesis loop
        raise NotImplementedError

    async def run(self) -> None:
        """Start all pipeline stages as concurrent tasks and run until cancelled."""
        logger.info("Pipeline starting.")
        # TODO: Issue #8 — gather mic, stt, llm, tts, and speaker tasks
        # Handle asyncio.CancelledError for graceful shutdown on Ctrl+C.
        raise NotImplementedError
