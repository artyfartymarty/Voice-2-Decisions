import io
import os

import numpy as np
import pytest
import soundfile as sf

from copilot.asr import FakeASR, set_backend


def pytest_collection_modifyitems(config, items):
    if os.environ.get("RUN_GPU_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="set RUN_GPU_TESTS=1 to run the real Nemotron model")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def fake_asr(request):
    """Every test uses FakeASR unless it is marked gpu."""
    if "gpu" in request.keywords:
        set_backend(None)
        yield None
    else:
        backend = FakeASR()
        set_backend(backend)
        yield backend
    set_backend(None)


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
