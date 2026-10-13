"""Demo mode without a GPU: transcripts and speakers come from scripts that sit next to the test-call WAVs.

ASR_BACKEND=demo (and DIARIZER=demo, its default then) looks for `<name>.script.json` beside each WAV in
TEST_CALLS_DIR: a list of lines {"start", "end", "speaker": "agent" | "caller", "text"} in seconds. When audio
arrives, whole or as a live chunk, it is located inside one of those WAVs by cross-correlation, and the backend
returns the words spoken in that stretch, so each chunk gets its own text, timed to the audio. Speakers come
back as "speaker 1" / "speaker 2" in order of first appearance, so role inference runs as it does with the
real diarizer.

Audio that matches no script transcribes to nothing (and is logged): demo mode never invents text.
scripts/make_demo_calls.py writes the demo calls and their scripts.
"""
import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from django.conf import settings
from scipy.signal import fftconvolve

from . import audio
from .asr import ASRBackend, Segment, Transcript
from .audio import TARGET_SR
from .diarize import Turn, speaker_label

log = logging.getLogger(__name__)
MIN_MATCH = 0.6  # normalized cross-correlation needed to call a chunk part of a script's audio
MIN_RMS = 1e-3   # quieter than this is silence: nothing to locate


@dataclass
class Word:
    start: float
    end: float
    speaker: str
    text: str


class Script:
    def __init__(self, wav: Path, lines: list[dict]):
        self.name = wav.name
        self.samples = audio.decode(wav.read_bytes())
        self.lines = sorted(lines, key=lambda l: l["start"])
        self.words = [w for line in self.lines for w in _words(line)]
        order = list(dict.fromkeys(l["speaker"] for l in self.lines))
        self.labels = {role: speaker_label(i) for i, role in enumerate(order)}  # agent -> "speaker 1", ...
        sq = np.concatenate([[0.0], np.cumsum(self.samples.astype(np.float64) ** 2)])
        self._energy = sq  # prefix sums of squares, for normalizing the correlation

    @property
    def duration(self) -> float:
        return len(self.samples) / TARGET_SR

    def ncc(self, x: np.ndarray) -> np.ndarray:
        """Normalized cross-correlation of x against every offset of the script audio."""
        n = len(x)
        corr = fftconvolve(self.samples, x[::-1], mode="valid")
        energy = self._energy[n:] - self._energy[:-n]
        return corr / (np.sqrt(np.maximum(energy, 1e-12)) * np.sqrt(float(np.dot(x, x))))

    def words_between(self, a: float, b: float) -> list[Word]:
        return [w for w in self.words if a <= (w.start + w.end) / 2 < b]

    def turns_until(self, t: float) -> list[Turn]:
        return [Turn(l["start"], min(l["end"], t), self.labels[l["speaker"]]) for l in self.lines if l["start"] < t]


def _words(line: dict) -> list[Word]:
    """Spread a line's words over its time span in proportion to their length."""
    tokens = line["text"].split()
    if not tokens:
        return []
    total = sum(len(t) + 1 for t in tokens)
    span = line["end"] - line["start"]
    out, t = [], line["start"]
    for tok in tokens:
        d = span * (len(tok) + 1) / total
        out.append(Word(t, t + d, line["speaker"], tok))
        t += d
    return out


class Library:
    """Every scripted WAV in TEST_CALLS_DIR, loaded once, and a locator for audio cut from them."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.scripts: list[Script] = []
        for path in sorted(folder.glob("*.script.json")) if folder.is_dir() else []:
            wav = path.with_name(path.name.removesuffix(".script.json") + ".wav")
            if wav.exists():
                self.scripts.append(Script(wav, json.loads(path.read_text())))
        self._last: tuple[Script, float] | None = None  # where the previous chunk ended: checked first
        self._lock = threading.Lock()

    def locate(self, samples: np.ndarray) -> tuple[Script, float] | None:
        """(script, start time in it) for a stretch of audio cut from a scripted WAV, or None."""
        x = np.asarray(samples, dtype=np.float32)
        if not len(x) or float(np.sqrt(np.mean(x ** 2))) < MIN_RMS:
            return None
        with self._lock:
            if self._last is not None:  # live calls arrive in order: try right after the last chunk first
                script, t = self._last
                hit = self._near(script, x, t)
                if hit is not None:
                    self._last = (script, hit + len(x) / TARGET_SR)
                    return script, hit
            best = None
            for script in self.scripts:
                if len(x) > len(script.samples):
                    continue
                ncc = script.ncc(x)
                i = int(np.argmax(ncc))
                if ncc[i] >= MIN_MATCH and (best is None or ncc[i] > best[2]):
                    best = (script, i / TARGET_SR, float(ncc[i]))
            if best is None:
                log.warning("demo ASR: %.1f s of audio matches no script in %s", len(x) / TARGET_SR, self.folder)
                return None
            self._last = (best[0], best[1] + len(x) / TARGET_SR)
            return best[0], best[1]

    @staticmethod
    def _near(script: Script, x: np.ndarray, t: float, slack: float = 0.5) -> float | None:
        a = max(0, int((t - slack) * TARGET_SR))
        b = min(len(script.samples), int((t + slack) * TARGET_SR) + len(x))
        if b - a < len(x):
            return None
        window = script.samples[a:b]
        corr = fftconvolve(window, x[::-1], mode="valid")
        sq = np.concatenate([[0.0], np.cumsum(window.astype(np.float64) ** 2)])
        energy = sq[len(x):] - sq[:-len(x)]
        ncc = corr / (np.sqrt(np.maximum(energy, 1e-12)) * np.sqrt(float(np.dot(x, x))))
        i = int(np.argmax(ncc))
        return (a + i) / TARGET_SR if ncc[i] >= MIN_MATCH else None


_library: Library | None = None
_library_lock = threading.Lock()


def library() -> Library:
    global _library
    with _library_lock:
        if _library is None or _library.folder != settings.TEST_CALLS_DIR:
            _library = Library(settings.TEST_CALLS_DIR)
        return _library


class DemoASR(ASRBackend):
    """Returns the scripted words for the stretch of a demo call it is given (see the module docstring)."""

    name = "demo"

    def transcribe(self, samples: np.ndarray) -> Transcript:
        t = Transcript(duration=len(samples) / TARGET_SR, language="en-US")
        hit = library().locate(samples)
        if hit is None:
            return t
        script, offset = hit
        words = script.words_between(offset, offset + t.duration)
        prev = None
        for w in words:  # one segment per run of the same speaker's words
            if prev is not None and prev.speaker == w.speaker:
                seg = t.segments[-1]
                seg.end, seg.text = w.end - offset, f"{seg.text} {w.text}"
            else:
                t.segments.append(Segment(max(0.0, w.start - offset), w.end - offset, w.text))
            prev = w
        return t

    def info(self):
        lib = library()
        return {"backend": self.name, "scripts": [s.name for s in lib.scripts], "folder": str(lib.folder)}


class DemoSession:
    """Live diarization for a demo call: the script's turns up to how much audio has arrived."""

    def __init__(self):
        self.audio = np.zeros(0, dtype=np.float32)
        self.hit: tuple[Script, float] | None = None

    def push(self, samples: np.ndarray) -> list[Turn]:
        self.audio = np.concatenate([self.audio, np.asarray(samples, dtype=np.float32)])
        if self.hit is None:
            self.hit = library().locate(self.audio)
        return self.turns()

    def finish(self) -> list[Turn]:
        return self.turns()

    def turns(self) -> list[Turn]:
        if self.hit is None:
            return []
        script, offset = self.hit  # call time 0 is `offset` into the script
        end = offset + len(self.audio) / TARGET_SR
        return [Turn(t.start - offset, t.end - offset, t.speaker) for t in script.turns_until(end) if t.end > offset]


class DemoDiarizer:
    name = "demo"

    def diarize(self, samples: np.ndarray) -> list[Turn]:
        s = DemoSession()
        s.push(samples)
        return s.finish()

    def session(self) -> DemoSession:
        return DemoSession()

    def load(self):
        library()

    def info(self):
        return {"backend": self.name}
