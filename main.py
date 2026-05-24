import asyncio
import logging

from agent.llm.openai import OpenAILLM
from agent.pipeline import Pipeline
from agent.stt.whisper import WhisperSTT
from agent.tts.elevenlabs import ElevenLabsTTS
from config.settings import Settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def build_pipeline(settings: Settings) -> Pipeline:
    stt = WhisperSTT(model_name=settings.whisper_model)
    llm = OpenAILLM(api_key=settings.openai_api_key, model=settings.llm_model)
    tts = ElevenLabsTTS(api_key=settings.elevenlabs_api_key, voice_id=settings.tts_voice_id)
    return Pipeline(settings=settings, stt=stt, llm=llm, tts=tts)


async def main() -> None:
    logger.info("Loading configuration...")
    settings = Settings.from_env()

    logger.info("Building pipeline...")
    pipeline = build_pipeline(settings)

    logger.info("Voice agent ready. Press Ctrl+C to stop.")
    await pipeline.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down.")
