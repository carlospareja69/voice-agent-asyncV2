"""
Tests for agent/stt/whisper.py

faster-whisper is mocked throughout — no model is downloaded, no GPU or heavy
ML dependency is required.  If the faster_whisper package is not installed the
module-level sys.modules stub ensures WhisperSTT can still be imported and
tested.

Test coverage
-------------
- Model is initialised with the correct arguments
- transcribe() reconstructs a float32 numpy array from the raw bytes
- transcribe() joins multiple segment texts with a single space
- transcribe() strips leading/trailing whitespace from the joined result
- transcribe() returns "" when there are no segments (silence)
- transcribe() runs the model in a thread-pool executor, not on the event loop
- transcribe() returns "" for empty (zero-length) audio bytes
"""

import sys
import threading
import unittest.mock
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Make the module importable even when faster-whisper is not installed.
# If the real package is present this line is a no-op; the real module wins.
# ---------------------------------------------------------------------------
if "faster_whisper" not in sys.modules:
    sys.modules["faster_whisper"] = unittest.mock.MagicMock()

from agent.stt.whisper import WhisperSTT  # noqa: E402 — must follow sys.modules patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_segment(text: str) -> MagicMock:
    """Return a mock faster-whisper Segment with the given text."""
    seg = MagicMock()
    seg.text = text
    return seg


def _make_stt(monkeypatch, model_name: str = "base") -> tuple["WhisperSTT", MagicMock]:
    """Return a (WhisperSTT, mock_model) pair with WhisperModel patched out."""
    mock_model = MagicMock()
    monkeypatch.setattr("agent.stt.whisper.WhisperModel", lambda *a, **kw: mock_model)
    stt = WhisperSTT(model_name=model_name)
    return stt, mock_model


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInit:
    def test_model_created_with_defaults(self, monkeypatch):
        """WhisperModel is called with model_name, device, and compute_type."""
        calls = []

        def fake_whisper_model(model_name, device, compute_type):
            calls.append((model_name, device, compute_type))
            return MagicMock()

        monkeypatch.setattr("agent.stt.whisper.WhisperModel", fake_whisper_model)
        WhisperSTT(model_name="base")

        assert calls == [("base", "cpu", "int8")]

    def test_model_created_with_custom_args(self, monkeypatch):
        """Custom device and compute_type are forwarded to WhisperModel."""
        calls = []

        def fake_whisper_model(model_name, device, compute_type):
            calls.append((model_name, device, compute_type))
            return MagicMock()

        monkeypatch.setattr("agent.stt.whisper.WhisperModel", fake_whisper_model)
        WhisperSTT(model_name="small", device="cuda", compute_type="float16")

        assert calls == [("small", "cuda", "float16")]

    def test_sample_rate_stored(self, monkeypatch):
        """Custom sample_rate is stored for downstream use."""
        stt, _ = _make_stt(monkeypatch)
        assert stt.sample_rate == 16000

        monkeypatch.setattr(
            "agent.stt.whisper.WhisperModel", lambda *a, **kw: MagicMock()
        )
        stt2 = WhisperSTT(sample_rate=8000)
        assert stt2.sample_rate == 8000


# ---------------------------------------------------------------------------
# transcribe() — output correctness
# ---------------------------------------------------------------------------


class TestTranscribe:
    async def test_single_segment_returned(self, monkeypatch):
        """A single segment produces a single stripped string."""
        stt, mock_model = _make_stt(monkeypatch)
        mock_model.transcribe.return_value = ([_make_segment(" Hello world")], MagicMock())

        result = await stt.transcribe(np.zeros(16000, dtype="float32").tobytes())

        assert result == "Hello world"

    async def test_multiple_segments_joined_with_space(self, monkeypatch):
        """Multiple segments are joined by a single space."""
        stt, mock_model = _make_stt(monkeypatch)
        segments = [_make_segment("Hello"), _make_segment("world")]
        mock_model.transcribe.return_value = (segments, MagicMock())

        result = await stt.transcribe(np.zeros(16000, dtype="float32").tobytes())

        assert result == "Hello world"

    async def test_empty_segments_returns_empty_string(self, monkeypatch):
        """No segments (silence) → empty string, not None or an exception."""
        stt, mock_model = _make_stt(monkeypatch)
        mock_model.transcribe.return_value = ([], MagicMock())

        result = await stt.transcribe(np.zeros(16000, dtype="float32").tobytes())

        assert result == ""

    async def test_leading_trailing_whitespace_stripped(self, monkeypatch):
        """Whisper often pads segment text; the final result is stripped."""
        stt, mock_model = _make_stt(monkeypatch)
        mock_model.transcribe.return_value = (
            [_make_segment("  padded text  ")],
            MagicMock(),
        )

        result = await stt.transcribe(np.zeros(16000, dtype="float32").tobytes())

        assert result == "padded text"

    async def test_empty_bytes_returns_empty_string(self, monkeypatch):
        """Zero-length audio bytes produce an empty transcript without crashing."""
        stt, mock_model = _make_stt(monkeypatch)
        mock_model.transcribe.return_value = ([], MagicMock())

        result = await stt.transcribe(b"")

        assert result == ""

    async def test_whitespace_only_segments_return_empty_string(self, monkeypatch):
        """Segments containing only whitespace collapse to an empty string."""
        stt, mock_model = _make_stt(monkeypatch)
        mock_model.transcribe.return_value = (
            [_make_segment("   "), _make_segment("  ")],
            MagicMock(),
        )

        result = await stt.transcribe(np.zeros(1024, dtype="float32").tobytes())

        assert result == ""


# ---------------------------------------------------------------------------
# transcribe() — audio reconstruction
# ---------------------------------------------------------------------------


class TestAudioReconstruction:
    async def test_model_receives_float32_numpy_array(self, monkeypatch):
        """transcribe() passes a float32 numpy array to model.transcribe, not bytes."""
        stt, mock_model = _make_stt(monkeypatch)
        mock_model.transcribe.return_value = ([], MagicMock())

        # Create known audio data
        original = np.linspace(-1.0, 1.0, 512, dtype=np.float32)
        audio_bytes = original.tobytes()

        await stt.transcribe(audio_bytes)

        # Inspect what the model actually received
        args, kwargs = mock_model.transcribe.call_args
        received_array = args[0]

        assert isinstance(received_array, np.ndarray)
        assert received_array.dtype == np.float32
        np.testing.assert_array_equal(received_array, original)

    async def test_bytes_correctly_round_trip_to_float32(self, monkeypatch):
        """np.frombuffer correctly reconstructs the original float32 values."""
        stt, mock_model = _make_stt(monkeypatch)
        mock_model.transcribe.return_value = ([], MagicMock())

        # Values that are exactly representable as float32
        original = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        await stt.transcribe(original.tobytes())

        received = mock_model.transcribe.call_args[0][0]
        np.testing.assert_array_equal(received, original)


# ---------------------------------------------------------------------------
# transcribe() — executor / threading
# ---------------------------------------------------------------------------


class TestExecutor:
    async def test_model_called_from_worker_thread(self, monkeypatch):
        """Inference runs in a thread-pool worker, not on the event loop thread."""
        stt, mock_model = _make_stt(monkeypatch)

        event_loop_thread = threading.current_thread()
        inference_thread: list[threading.Thread] = []

        def fake_transcribe(audio, **kwargs):
            inference_thread.append(threading.current_thread())
            return [], MagicMock()

        mock_model.transcribe.side_effect = fake_transcribe

        await stt.transcribe(np.zeros(1024, dtype="float32").tobytes())

        assert len(inference_thread) == 1
        # The model must NOT have been called on the event loop thread.
        assert inference_thread[0] is not event_loop_thread

    async def test_transcribe_does_not_block_event_loop(self, monkeypatch):
        """A concurrent asyncio task can run while transcription is in progress."""
        stt, mock_model = _make_stt(monkeypatch)

        # Simulate slow inference using a threading.Event
        inference_started = threading.Event()
        inference_may_finish = threading.Event()

        def slow_transcribe(audio, **kwargs):
            inference_started.set()
            inference_may_finish.wait(timeout=5)
            return [], MagicMock()

        mock_model.transcribe.side_effect = slow_transcribe

        side_task_ran = []

        async def side_task():
            side_task_ran.append(True)

        import asyncio

        audio = np.zeros(1024, dtype="float32").tobytes()
        transcribe_task = asyncio.create_task(stt.transcribe(audio))

        # Wait until the worker thread has actually started
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: inference_started.wait(timeout=5)
        )

        # The side task can run while transcription is blocked in the thread
        await asyncio.create_task(side_task())

        # Unblock the worker thread
        inference_may_finish.set()
        await transcribe_task

        assert side_task_ran == [True], "Event loop was blocked during transcription"
