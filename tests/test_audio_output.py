"""
Tests for agent/audio/output.py

No real speaker or PortAudio device is required.  sounddevice.OutputStream
is replaced by FakeOutputStream in every test, so the suite runs in any CI
environment.

Test coverage
-------------
- Default and custom constructor values
- OutputStream is opened with samplerate, channels=1, dtype="int16"
- Audio bytes placed in the asyncio queue fill outdata via the callback
- All zeros (silence) are written when the asyncio queue is empty
- Partial audio shorter than one callback frame is zero-padded
- A chunk that spans two callbacks is split correctly (leftover continuity)
- CancelledError propagates out of stream() so callers see shutdown
- A non-None status flag triggers a WARNING log entry
"""

import asyncio
import logging

import numpy as np
import pytest

from agent.audio.output import SpeakerOutput


# ---------------------------------------------------------------------------
# Fake sounddevice.OutputStream
# ---------------------------------------------------------------------------


class FakeOutputStream:
    """Minimal stand-in for sounddevice.OutputStream.

    Stores every keyword argument so tests can assert what the real stream
    would have been opened with.  Exposes ``callback`` so tests can fire it
    manually to simulate PortAudio's audio thread.
    """

    def __init__(self, *, callback, samplerate, channels, dtype, **kwargs):
        self.callback = callback
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype

    def __enter__(self) -> "FakeOutputStream":
        return self

    def __exit__(self, *_) -> None:
        pass


@pytest.fixture()
def fake_stream(monkeypatch) -> list[FakeOutputStream]:
    """Patch sounddevice.OutputStream with FakeOutputStream.

    Returns a list that is populated with each instance created during the
    test.  Most tests only create one stream, so they read ``instances[0]``.
    """
    instances: list[FakeOutputStream] = []

    class CapturingFakeOutputStream(FakeOutputStream):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            instances.append(self)

    monkeypatch.setattr(
        "agent.audio.output.sounddevice.OutputStream",
        CapturingFakeOutputStream,
    )
    return instances


# ---------------------------------------------------------------------------
# Helper: start stream(), wait for the OutputStream to be created, then yield
# ---------------------------------------------------------------------------


async def _start_stream(
    speaker: SpeakerOutput, queue: asyncio.Queue[bytes]
) -> asyncio.Task:
    """Create a stream task and wait one event-loop tick for it to start.

    After this returns, the FakeOutputStream has been entered and its
    ``callback`` is available via ``fake_stream[0].callback``.  The task is
    suspended at ``await queue.get()``.
    """
    task = asyncio.create_task(speaker.stream(queue))
    # Yield once so the task runs up to `await queue.get()` and the
    # FakeOutputStream context manager has been entered.
    await asyncio.sleep(0)
    return task


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestInit:
    def test_default_sample_rate(self):
        """Default sample_rate is 24000 Hz — matches ElevenLabsTTS pcm_24000."""
        speaker = SpeakerOutput()
        assert speaker.sample_rate == 24000

    def test_custom_sample_rate(self):
        """A custom sample_rate is stored and forwarded to OutputStream."""
        speaker = SpeakerOutput(sample_rate=44100)
        assert speaker.sample_rate == 44100


# ---------------------------------------------------------------------------
# stream() behavioural tests
# ---------------------------------------------------------------------------


class TestStream:
    async def test_stream_opens_with_correct_parameters(self, fake_stream):
        """OutputStream is created with sample_rate, channels=1, dtype='int16'."""
        speaker = SpeakerOutput(sample_rate=44100)
        queue: asyncio.Queue[bytes] = asyncio.Queue()

        task = await _start_stream(speaker, queue)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert len(fake_stream) == 1
        stream = fake_stream[0]
        assert stream.samplerate == 44100
        assert stream.channels == 1
        assert stream.dtype == "int16"

    async def test_audio_fills_outdata_from_queue(self, fake_stream):
        """PCM bytes queued via asyncio land in outdata through the callback."""
        speaker = SpeakerOutput()
        queue: asyncio.Queue[bytes] = asyncio.Queue()

        task = await _start_stream(speaker, queue)

        # 4 int16 samples = 8 bytes
        samples = np.array([10, 20, 30, 40], dtype=np.int16)
        queue.put_nowait(samples.tobytes())

        # Allow the task one tick to dequeue and forward to the stdlib queue.
        await asyncio.sleep(0)

        outdata = np.zeros((4, 1), dtype=np.int16)
        fake_stream[0].callback(outdata, 4, None, None)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        np.testing.assert_array_equal(outdata[:, 0], samples)

    async def test_silence_when_queue_empty(self, fake_stream):
        """outdata is filled with zeros when no audio is available."""
        speaker = SpeakerOutput()
        queue: asyncio.Queue[bytes] = asyncio.Queue()

        task = await _start_stream(speaker, queue)

        # No data put in queue — callback should write silence.
        outdata = np.ones((4, 1), dtype=np.int16)  # pre-filled with non-zero
        fake_stream[0].callback(outdata, 4, None, None)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        np.testing.assert_array_equal(outdata[:, 0], np.zeros(4, dtype=np.int16))

    async def test_partial_chunk_silence_padded(self, fake_stream):
        """When audio is shorter than one frame the remainder is zero-padded."""
        speaker = SpeakerOutput()
        queue: asyncio.Queue[bytes] = asyncio.Queue()

        task = await _start_stream(speaker, queue)

        # Only 2 int16 samples, but callback requests 4 frames.
        samples = np.array([100, 200], dtype=np.int16)
        queue.put_nowait(samples.tobytes())
        await asyncio.sleep(0)

        outdata = np.ones((4, 1), dtype=np.int16)
        fake_stream[0].callback(outdata, 4, None, None)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # First 2 frames filled, last 2 padded with silence.
        assert outdata[0, 0] == 100
        assert outdata[1, 0] == 200
        assert outdata[2, 0] == 0
        assert outdata[3, 0] == 0

    async def test_cross_callback_leftover_continuity(self, fake_stream):
        """Audio bytes spanning two callbacks are split correctly via leftover."""
        speaker = SpeakerOutput()
        queue: asyncio.Queue[bytes] = asyncio.Queue()

        task = await _start_stream(speaker, queue)

        # 6 int16 samples = 12 bytes; callbacks request 4 frames each.
        samples = np.array([1, 2, 3, 4, 5, 6], dtype=np.int16)
        queue.put_nowait(samples.tobytes())
        await asyncio.sleep(0)

        outdata1 = np.zeros((4, 1), dtype=np.int16)
        fake_stream[0].callback(outdata1, 4, None, None)

        outdata2 = np.zeros((4, 1), dtype=np.int16)
        fake_stream[0].callback(outdata2, 4, None, None)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # First callback: samples 1–4
        np.testing.assert_array_equal(outdata1[:, 0], [1, 2, 3, 4])
        # Second callback: samples 5–6 then silence
        assert outdata2[0, 0] == 5
        assert outdata2[1, 0] == 6
        assert outdata2[2, 0] == 0
        assert outdata2[3, 0] == 0

    async def test_cancelled_error_propagates(self, fake_stream):
        """stream() re-raises CancelledError so the pipeline task sees shutdown."""
        speaker = SpeakerOutput()
        queue: asyncio.Queue[bytes] = asyncio.Queue()

        task = await _start_stream(speaker, queue)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_status_flag_emits_warning(self, fake_stream, caplog):
        """A non-None status from PortAudio is logged at WARNING level."""
        speaker = SpeakerOutput()
        queue: asyncio.Queue[bytes] = asyncio.Queue()

        task = await _start_stream(speaker, queue)

        outdata = np.zeros((4, 1), dtype=np.int16)
        with caplog.at_level(logging.WARNING, logger="agent.audio.output"):
            fake_stream[0].callback(outdata, 4, None, "output_underflow")

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("output_underflow" in msg for msg in warning_messages)
