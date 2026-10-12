import shutil

import numpy as np
import pytest

from copilot import audio
from tests.conftest import tone, wav_bytes


def test_decode_wav_is_mono_16k():
    out = audio.decode(wav_bytes(tone(1.0, sr=44100, channels=2), sr=44100))
    assert out.dtype == np.float32 and out.ndim == 1
    assert abs(len(out) - 16000) <= 1


def test_decode_empty_raises():
    with pytest.raises(audio.AudioDecodeError):
        audio.decode(b"")


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_decode_garbage_raises_via_ffmpeg():
    with pytest.raises(audio.AudioDecodeError):
        audio.decode(b"definitely not audio" * 100)


def test_mix_legs_pads_and_never_clips():
    a = np.full(16000, 0.8, dtype=np.float32)
    b = np.full(8000, 0.8, dtype=np.float32)
    mixed = audio.mix_legs([a, b])
    assert len(mixed) == 16000
    assert np.abs(mixed).max() <= 1.0
    assert mixed[12000] < mixed[0]  # second half has only one leg


def test_mix_single_leg_unchanged():
    a = tone(0.5)
    assert np.allclose(audio.mix_legs([a]), a)


def test_split_on_silence_cuts_in_the_quiet_part():
    sr = 16000
    loud = tone(2.0, amp=0.5)
    quiet = np.zeros(sr // 2, dtype=np.float32)
    x = np.concatenate([loud, loud, quiet, loud])  # quiet gap sits at 4.0-4.5 s
    spans = audio.split_on_silence(x, sr, max_seconds=5.0, search_seconds=2.0)
    assert spans[0][0] == 0 and spans[-1][1] == len(x)
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))  # contiguous
    assert 4.0 * sr <= spans[0][1] <= 4.5 * sr


def test_split_short_audio_is_one_span():
    assert audio.split_on_silence(np.zeros(100, dtype=np.float32)) == [(0, 100)]
