"""
Tests for config/settings.py

These tests exercise validation logic only.
They never read from a real .env file and never call any external API.
"""

import os
import pytest

from config.settings import Settings, _get_int, _require, VALID_WHISPER_MODELS


# ---------------------------------------------------------------------------
# Helper: build a minimal valid Settings object without touching the env
# ---------------------------------------------------------------------------

def _make_settings(**overrides) -> Settings:
    """Return a Settings with dummy API keys, overriding any field via kwargs."""
    defaults = dict(
        openai_api_key="sk-test-openai",
        elevenlabs_api_key="el-test-key",
    )
    defaults.update(overrides)
    return Settings(**defaults)


# ---------------------------------------------------------------------------
# _require()
# ---------------------------------------------------------------------------

class TestRequire:
    def test_returns_value_when_set(self, monkeypatch):
        monkeypatch.setenv("SOME_KEY", "hello")
        assert _require("SOME_KEY") == "hello"

    def test_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("SOME_KEY", "  hello  ")
        assert _require("SOME_KEY") == "hello"

    def test_raises_when_missing(self, monkeypatch):
        monkeypatch.delenv("SOME_KEY", raising=False)
        with pytest.raises(ValueError, match="SOME_KEY"):
            _require("SOME_KEY")

    def test_raises_when_blank(self, monkeypatch):
        monkeypatch.setenv("SOME_KEY", "   ")
        with pytest.raises(ValueError, match="SOME_KEY"):
            _require("SOME_KEY")


# ---------------------------------------------------------------------------
# _get_int()
# ---------------------------------------------------------------------------

class TestGetInt:
    def test_returns_default_when_not_set(self, monkeypatch):
        monkeypatch.delenv("SOME_INT", raising=False)
        assert _get_int("SOME_INT", 42) == 42

    def test_parses_valid_integer(self, monkeypatch):
        monkeypatch.setenv("SOME_INT", "8000")
        assert _get_int("SOME_INT", 0) == 8000

    def test_raises_on_non_integer(self, monkeypatch):
        monkeypatch.setenv("SOME_INT", "abc")
        with pytest.raises(ValueError, match="SOME_INT"):
            _get_int("SOME_INT", 0)

    def test_raises_on_zero(self, monkeypatch):
        monkeypatch.setenv("SOME_INT", "0")
        with pytest.raises(ValueError, match="positive"):
            _get_int("SOME_INT", 0)

    def test_raises_on_negative(self, monkeypatch):
        monkeypatch.setenv("SOME_INT", "-1")
        with pytest.raises(ValueError, match="positive"):
            _get_int("SOME_INT", 0)


# ---------------------------------------------------------------------------
# Settings dataclass
# ---------------------------------------------------------------------------

class TestSettings:
    def test_valid_settings_object(self):
        s = _make_settings()
        assert s.llm_model == "gpt-4o-mini"
        assert s.whisper_model == "base"
        assert s.audio_sample_rate == 16000
        assert s.audio_chunk_size == 1024
        assert s.tts_sample_rate == 24000

    def test_tts_sample_rate_default(self):
        """Default tts_sample_rate is 24000 Hz — matches pcm_24000 from ElevenLabs."""
        s = _make_settings()
        assert s.tts_sample_rate == 24000

    def test_api_keys_hidden_in_repr(self):
        s = _make_settings()
        r = repr(s)
        assert "sk-test-openai" not in r
        assert "el-test-key" not in r

    def test_invalid_whisper_model_raises(self):
        with pytest.raises(ValueError, match="WHISPER_MODEL"):
            _make_settings(whisper_model="not-a-real-model")

    def test_all_valid_whisper_models_accepted(self):
        for model in VALID_WHISPER_MODELS:
            s = _make_settings(whisper_model=model)
            assert s.whisper_model == model

    def test_custom_optional_fields(self):
        s = _make_settings(
            llm_model="gpt-4o",
            tts_voice_id="custom-voice",
            audio_sample_rate=8000,
            audio_chunk_size=512,
        )
        assert s.llm_model == "gpt-4o"
        assert s.tts_voice_id == "custom-voice"
        assert s.audio_sample_rate == 8000
        assert s.audio_chunk_size == 512


# ---------------------------------------------------------------------------
# Settings.from_env()
# ---------------------------------------------------------------------------

class TestFromEnv:
    def test_loads_required_keys(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
        monkeypatch.setenv("ELEVENLABS_API_KEY", "el-real")
        # Unset optionals so defaults are used
        for key in ("WHISPER_MODEL", "LLM_MODEL", "TTS_VOICE_ID",
                    "AUDIO_SAMPLE_RATE", "AUDIO_CHUNK_SIZE", "TTS_SAMPLE_RATE"):
            monkeypatch.delenv(key, raising=False)

        s = Settings.from_env()
        assert s.llm_model == "gpt-4o-mini"
        assert s.whisper_model == "base"
        assert s.audio_sample_rate == 16000
        assert s.tts_sample_rate == 24000

    def test_raises_without_openai_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ELEVENLABS_API_KEY", "el-real")
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            Settings.from_env()

    def test_raises_without_elevenlabs_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        with pytest.raises(ValueError, match="ELEVENLABS_API_KEY"):
            Settings.from_env()

    def test_overrides_optional_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
        monkeypatch.setenv("ELEVENLABS_API_KEY", "el-real")
        monkeypatch.setenv("WHISPER_MODEL", "small")
        monkeypatch.setenv("AUDIO_SAMPLE_RATE", "8000")

        s = Settings.from_env()
        assert s.whisper_model == "small"
        assert s.audio_sample_rate == 8000

    def test_raises_on_bad_audio_sample_rate(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
        monkeypatch.setenv("ELEVENLABS_API_KEY", "el-real")
        monkeypatch.setenv("AUDIO_SAMPLE_RATE", "not-a-number")
        with pytest.raises(ValueError, match="AUDIO_SAMPLE_RATE"):
            Settings.from_env()

    def test_overrides_tts_sample_rate_from_env(self, monkeypatch):
        """TTS_SAMPLE_RATE env var overrides the 24000 default."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
        monkeypatch.setenv("ELEVENLABS_API_KEY", "el-real")
        monkeypatch.setenv("TTS_SAMPLE_RATE", "44100")
        for key in ("WHISPER_MODEL", "LLM_MODEL", "TTS_VOICE_ID",
                    "AUDIO_SAMPLE_RATE", "AUDIO_CHUNK_SIZE"):
            monkeypatch.delenv(key, raising=False)

        s = Settings.from_env()
        assert s.tts_sample_rate == 44100

    def test_raises_on_bad_tts_sample_rate(self, monkeypatch):
        """A non-integer TTS_SAMPLE_RATE raises a clear ValueError."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
        monkeypatch.setenv("ELEVENLABS_API_KEY", "el-real")
        monkeypatch.setenv("TTS_SAMPLE_RATE", "abc")
        with pytest.raises(ValueError, match="TTS_SAMPLE_RATE"):
            Settings.from_env()
