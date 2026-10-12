"""Audio decoding and mixing.

Everything downstream works on mono float32 at 16 kHz, the rate Nemotron ASR expects.
A call usually has two legs (agent and caller); mix_legs() folds them into one mono leg.
"""
import io
import shutil
import subprocess
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000


class AudioDecodeError(ValueError):
    pass


def to_mono(samples: np.ndarray) -> np.ndarray:
    """(frames,) or (frames, channels) -> (frames,) by averaging channels."""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    return samples


def resample(samples: np.ndarray, sr_from: int, sr_to: int = TARGET_SR) -> np.ndarray:
    if sr_from == sr_to:
        return samples.astype(np.float32, copy=False)
    g = gcd(sr_from, sr_to)
    return resample_poly(samples, sr_to // g, sr_from // g).astype(np.float32)


def mix_legs(legs: list[np.ndarray]) -> np.ndarray:
    """Sum mono legs of possibly different lengths into one leg, scaled down only if it would clip."""
    legs = [to_mono(leg) for leg in legs if leg is not None and len(leg)]
    if not legs:
        return np.zeros(0, dtype=np.float32)
    mixed = np.zeros(max(len(leg) for leg in legs), dtype=np.float32)
    for leg in legs:
        mixed[: len(leg)] += leg
    peak = float(np.abs(mixed).max()) if len(mixed) else 0.0
    if peak > 1.0:
        mixed /= peak
    return mixed


def _decode_with_ffmpeg(data: bytes) -> np.ndarray:
    if not shutil.which("ffmpeg"):
        raise AudioDecodeError("format not readable by libsndfile and ffmpeg is not installed")
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", "pipe:0",
         "-f", "f32le", "-ac", "1", "-ar", str(TARGET_SR), "pipe:1"],
        input=data, capture_output=True, check=False,
    )
    if proc.returncode != 0:
        raise AudioDecodeError(f"ffmpeg could not decode audio: {proc.stderr.decode(errors='replace')[:300]}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def decode(data: bytes) -> np.ndarray:
    """Any audio file's bytes -> mono float32 at 16 kHz. WAV/FLAC/OGG via libsndfile, the rest (mp3, webm, m4a) via ffmpeg."""
    if not data:
        raise AudioDecodeError("empty audio")
    try:
        samples, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception:
        return _decode_with_ffmpeg(data)
    return resample(to_mono(samples), sr)


def split_on_silence(samples: np.ndarray, sr: int = TARGET_SR, max_seconds: float = 30.0,
                     search_seconds: float = 3.0, frame_ms: int = 20) -> list[tuple[int, int]]:
    """Cut long audio into chunks of at most max_seconds, preferring the quietest frame near each cut.

    Returns (start_sample, end_sample) pairs covering the whole input with no gaps.
    """
    n = len(samples)
    max_len = int(max_seconds * sr)
    if n <= max_len:
        return [(0, n)] if n else []
    frame = max(1, int(sr * frame_ms / 1000))
    search = int(search_seconds * sr)
    spans, start = [], 0
    while n - start > max_len:
        hard_end = start + max_len
        lo = max(start + 1, hard_end - search)
        window = samples[lo:hard_end]
        usable = len(window) // frame * frame
        if usable:
            energy = (window[:usable].reshape(-1, frame) ** 2).mean(axis=1)
            cut = lo + int(energy.argmin()) * frame + frame // 2
        else:
            cut = hard_end
        spans.append((start, cut))
        start = cut
    spans.append((start, n))
    return spans
