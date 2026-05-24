from .base import STTProvider


class WhisperSTT(STTProvider):
    """Local speech-to-text using faster-whisper."""

    def __init__(self, model_name: str = "base") -> None:
        self.model_name = model_name
        # TODO: Issue #4 — load faster-whisper model
        # self.model = WhisperModel(model_name, device="cpu", compute_type="int8")

    async def transcribe(self, audio: bytes) -> str:
        # TODO: Issue #4 — run model in a thread-pool executor to avoid blocking
        # the event loop (faster-whisper is synchronous/CPU-bound).
        raise NotImplementedError
