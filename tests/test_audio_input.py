"""
Tests for agent/audio/input.py

No real microphone or PortAudio device is required.  sounddevice.InputStream
is replaced by FakeInputStream in every test, so the suite runs in any CI
environment.

Test coverage
-------------
- Default and custom constructor values
- Audio bytes land in the queue via call_soon_threadsafe
- Bytes content matches the raw float32 PCM data
- Multiple successive callbacks each produce one queue entry
- CancelledError propagates out of stream() so callers see shutdown
- InputStream is opened with the parameters from MicrophoneInput
- A non-None status flag triggers a WARNING log entry
"""

import asyncio
import logging

import numpy as np
import pytest

from agent.audio.input import MicrophoneInput


# ---------------------------------------------------------------------------
# Fake sounddevice.InputStream
# ---------------------------------------------------------------------------


class FakeInputStream:
    """Minimal stand-in for sounddevice.InputStream.

    Stores every keyword argument so tests can assert what the real stream
    would have been opened with.  Exposes ``callback`` so tests can fire it
    manually to simulate PortAudio's audio thread.
    """

    def __init__(self, *, callback, samplerate, blocksize, channels, dtype):
        self.callback = callback
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.channels = channels
        self.dtype = dtype

    def __enter__(self) -> "FakeInputStream":
        return self

    def __exit__(self, *_) -> None:
        pass


@pytest.fixture()
def fake_stream(monkeypatch) -> list[FakeInputStream]:
    """Patch sounddevice.InputStream with FakeInputStream.

    Returns a list that is populated with each instance created during the
    test.  Most tests only create one stream, so they read ``instances[0]``.
    """
    instances: list[FakeInputStream] = []

    class CapturingFakeInputStream(FakeInputStream):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            instances.append(self)

    monkeypatch.setattr(
        "agent.audio.input.sounddevice.InputStream",
        CapturingFakeInputStream,
    )
    return instances


# ---------------------------------------------------------------------------
# Helper: start stream(), wait for the InputStream to be created, then yield
# ---------------------------------------------------------------------------


async def _start_stream(mic: MicrophoneInput, queue: asyncio.Queue[bytes]):
    """Create a stream task and wait one event-loop tick for it to start."""
    task = asyncio.create_task(mic.stream(queue))
    # Yield once so the task runs up to `await asyncio.Event().wait()` and the
    # FakeInputStream context manager has been entered.
    await asyncio.sleep(0)
    return task


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestInit:
    def test_default_sample_rate_and_chunk_size(self):
        mic = MicrophoneInput()
        assert mic.sample_rate == 16000
        assert mic.chunk_size == 1024

    def test_custom_sample_rate_and_chunk_size(self):
        mic = MicrophoneInput(sample_rate=8000, chunk_size=512)
        assert mic.sample_rate == 8000
        assert mic.chunk_size == 512


# ---------------------------------------------------------------------------
# stream() behavioural tests
# ---------------------------------------------------------------------------


class TestStream:
    async def test_audio_bytes_land_in_queue(self, fake_stream):
        """A single callback invocation enqueues exactly one bytes item."""
        mic = MicrophoneInput(sample_rate=16000, chunk_size=4)
        queue: asyncio.Queue[bytes] = asyncio.Queue()

        task = await _start_stream(mic, queue)

        fake_audio = np.ones((4, 1), dtype="float32")
        fake_stream[0].callback(fake_audio, 4, None, None)

        # Allow call_soon_threadsafe to deliver the put_nowait to the loop.
        await asyncio.sleep(0.05)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert queue.qsize() == 1

    async def test_enqueued_bytes_match_raw_pcm(self, fake_stream):
        """The bytes in the queue are identical to indata.tobytes()."""
        mic = MicrophoneInput(sample_rate=16000, chunk_size=4)
        queue: asyncio.Queue[bytes] = asyncio.Queue()

        task = await _start_stream(mic, queue)

        fake_audio = np.full((4, 1), 0.5, dtype="float32")
        fake_stream[0].callback(fake_audio, 4, None, None)
        await asyncio.sleep(0.05)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        chunk = queue.get_nowait()
        assert isinstance(chunk, bytes)
        assert chunk == fake_audio.tobytes()

    async def test_multiple_callbacks_produce_multiple_queue_entries(self, fake_stream):
        """Each callback invocation appends a separate entry to the queue."""
        mic = MicrophoneInput(sample_rate=16000, chunk_size=4)
        queue: asyncio.Queue[bytes] = asyncio.Queue()

        task = await _start_stream(mic, queue)

        for value in (0.1, 0.2, 0.3):
            audio = np.full((4, 1), value, dtype="float32")
            fake_stream[0].callback(audio, 4, None, None)

        await asyncio.sleep(0.05)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert queue.qsize() == 3

    async def test_cancelled_error_propagates_to_caller(self, fake_stream):
        """stream() re-raises CancelledError so the pipeline task sees shutdown."""
        mic = MicrophoneInput()
        queue: asyncio.Queue[bytes] = asyncio.Queue()

        task = await _start_stream(mic, queue)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_stream_opens_with_correct_parameters(self, fake_stream):
        """InputStream is created with sample_rate, chunk_size, mono, float32."""
        mic = MicrophoneInput(sample_rate=8000, chunk_size=512)
        queue: asyncio.Queue[bytes] = asyncio.Queue()

        task = await _start_stream(mic, queue)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert len(fake_stream) == 1
        stream = fake_stream[0]
        assert stream.samplerate == 8000
        assert stream.blocksize == 512
        assert stream.channels == 1
        assert stream.dtype == "float32"

    async def test_status_flag_emits_warning(self, fake_stream, caplog):
        """A non-None status from PortAudio is logged at WARNING level."""
        mic = MicrophoneInput(sample_rate=16000, chunk_size=4)
        queue: asyncio.Queue[bytes] = asyncio.Queue()

        task = await _start_stream(mic, queue)

        fake_audio = np.zeros((4, 1), dtype="float32")
        with caplog.at_level(logging.WARNING, logger="agent.audio.input"):
            fake_stream[0].callback(fake_audio, 4, None, "input_overflow")
            await asyncio.sleep(0.05)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("input_overflow" in msg for msg in warning_messages)

    async def test_chunk_is_independent_copy_of_indata(self, fake_stream):
        """Mutating indata after the callback does not corrupt the queued bytes."""
        mic = MicrophoneInput(sample_rate=16000, chunk_size=4)
        queue: asyncio.Queue[bytes] = asyncio.Queue()

        task = await _start_stream(mic, queue)

        fake_audio = np.ones((4, 1), dtype="float32")
        expected_bytes = fake_audio.tobytes()

        fake_stream[0].callback(fake_audio, 4, None, None)
        # Mutate the array in-place — simulates PortAudio reusing its buffer.
        fake_audio[:] = 0.0

        await asyncio.sleep(0.05)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        chunk = queue.get_nowait()
        # The queued bytes must reflect the original values, not the mutation.
        assert chunk == expected_bytes
