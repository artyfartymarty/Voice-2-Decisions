import io
import os

import numpy as np
import pytest
import soundfile as sf

from copilot.asr import FakeASR, set_backend
from copilot.diarize import FakeDiarizer, set_diarizer


def pytest_collection_modifyitems(config, items):
    if os.environ.get("RUN_GPU_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="set RUN_GPU_TESTS=1 to run the real Nemotron model")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def fake_asr(request, settings):
    """Every test uses FakeASR and no diarizer unless it is marked gpu."""
    set_diarizer(None)
    settings.LIVE_FLUSH_S = 0  # no background flush timers unless a test turns them on
    if "gpu" in request.keywords:
        set_backend(None)
        yield None
    else:
        settings.DIARIZER = "off"
        backend = FakeASR()
        set_backend(backend)
        yield backend
    set_backend(None)
    set_diarizer(None)


class ScriptedASR(FakeASR):
    """Returns the next line of a script on every transcribe() call (one per diarized turn or live chunk)."""

    def __init__(self, lines):
        super().__init__()
        self.lines = list(lines)

    def transcribe(self, samples):
        from copilot.audio import utterances
        if utterances(samples):  # silence consumes no line, like a real model returning nothing
            self.text = self.lines.pop(0) if self.lines else ""
        return super().transcribe(samples)


@pytest.fixture
def scripted(settings):
    """scripted(lines) -> installs ScriptedASR + FakeDiarizer (utterances alternate speaker 1 / speaker 2)."""
    def install(lines):
        settings.DIARIZER = "fake"
        set_diarizer(FakeDiarizer())
        asr = ScriptedASR(lines)
        set_backend(asr)
        return asr
    return install


def turns_audio(n, seconds=1.0, gap=1.0):
    """n tone 'utterances' separated by silence: what FakeDiarizer sees as n alternating turns."""
    silence = np.zeros(int(gap * 16000), dtype=np.float32)
    parts = [silence]
    for _ in range(n):
        parts += [tone(seconds), silence]
    return np.concatenate(parts)


def tone(seconds=1.0, sr=16000, freq=440.0, amp=0.3, channels=1):
    t = np.arange(int(seconds * sr)) / sr
    x = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.stack([x] * channels, axis=1) if channels > 1 else x


def wav_bytes(samples, sr=16000):
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.fixture
def wav():
    return wav_bytes(tone(1.0))
