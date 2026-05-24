import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


@dataclass
class Settings:
    openai_api_key: str
    elevenlabs_api_key: str
    whisper_model: str = "base"
    llm_model: str = "gpt-4o-mini"
    tts_voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    audio_sample_rate: int = 16000
    audio_chunk_size: int = 1024

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()

        openai_key = os.getenv("OPENAI_API_KEY", "")
        elevenlabs_key = os.getenv("ELEVENLABS_API_KEY", "")

        if not openai_key:
            raise ValueError("OPENAI_API_KEY is not set in your environment or .env file.")
        if not elevenlabs_key:
            raise ValueError("ELEVENLABS_API_KEY is not set in your environment or .env file.")

        return cls(
            openai_api_key=openai_key,
            elevenlabs_api_key=elevenlabs_key,
            whisper_model=os.getenv("WHISPER_MODEL", "base"),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            tts_voice_id=os.getenv("TTS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
        )
