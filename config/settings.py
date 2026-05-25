import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Valid faster-whisper model sizes.
# Reference: https://github.com/SYSTRAN/faster-whisper
VALID_WHISPER_MODELS = {
    "tiny",
    "base",
    "small",
    "medium",
    "large",
    "large-v2",
    "large-v3",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require(name: str) -> str:
    """Return the value of a required env variable, or raise a clear error."""
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(
            f"{name} is not set. "
            "Add it to your .env file or export it in your shell."
        )
    return value


def _get_int(name: str, default: int) -> int:
    """Return an env variable parsed as a positive integer, or a default value."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"{name} must be an integer, got: {raw!r}"
        )
    if value <= 0:
        raise ValueError(
            f"{name} must be a positive integer, got: {value}"
        )
    return value


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class Settings:
    """All runtime configuration for the voice agent.

    Never instantiate this class directly in production code.
    Always use Settings.from_env() so that .env is loaded and
    all values are validated before the pipeline starts.

    Example:
        settings = Settings.from_env()
        print(settings.llm_model)   # "gpt-4o-mini"
        print(settings)              # API keys are hidden (repr=False)
    """

    # --- Required fields (no repr so keys are never printed/logged) ---
    openai_api_key: str = field(repr=False)
    elevenlabs_api_key: str = field(repr=False)

    # --- Provider options ---
    whisper_model: str = "base"
    llm_model: str = "gpt-4o-mini"
    tts_voice_id: str = "21m00Tcm4TlvDq8ikWAM"

    # --- Audio settings ---
    audio_sample_rate: int = 16000   # Hz — must match the STT model's expected rate
    audio_chunk_size: int = 1024     # Frames per mic capture callback

    def __post_init__(self) -> None:
        """Validate field values immediately after the object is created."""
        if self.whisper_model not in VALID_WHISPER_MODELS:
            raise ValueError(
                f"WHISPER_MODEL {self.whisper_model!r} is not recognised. "
                f"Choose one of: {sorted(VALID_WHISPER_MODELS)}"
            )

    @classmethod
    def from_env(cls) -> "Settings":
        """Load configuration from the .env file and environment variables.

        Raises ValueError if any required variable is missing or any value
        fails validation.  Call this once at startup before building the
        pipeline.
        """
        load_dotenv()  # reads .env if present; real env vars always win

        return cls(
            # Required
            openai_api_key=_require("OPENAI_API_KEY"),
            elevenlabs_api_key=_require("ELEVENLABS_API_KEY"),
            # Optional with sensible defaults
            whisper_model=os.getenv("WHISPER_MODEL", "base"),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            tts_voice_id=os.getenv("TTS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
            audio_sample_rate=_get_int("AUDIO_SAMPLE_RATE", 16000),
            audio_chunk_size=_get_int("AUDIO_CHUNK_SIZE", 1024),
        )
