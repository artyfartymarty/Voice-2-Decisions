"""Who spoke when, on a single mono recording.

NemotronDiarizer runs nvidia/Nemotron-3-Diarization (Streaming Sortformer, ~100M params) through Transformers.
It always uses the model's streaming mode: a recording of any length is fed chunk by chunk with the speaker
cache carried along, and a live call keeps one StreamingSession open across HTTP requests so speaker
identities stay stable for the whole call.

Speakers come out as "speaker 1", "speaker 2"... in order of first appearance; roles.py names them agent/caller.
"""
import logging
import threading
from dataclasses import dataclass

import numpy as np
from django.conf import settings

from .audio import TARGET_SR, gpu_dtype, utterances

log = logging.getLogger(__name__)


@dataclass
class Turn:
    start: float
    end: float
    speaker: str


def speaker_label(index: int) -> str:
    return f"speaker {index + 1}"


def clean_turns(raw: list[Turn], merge_gap: float = 0.8, max_len: float = 30.0, min_len: float = 0.2) -> list[Turn]:
    """Sort, trim overlaps (later turn starts where the earlier one ends), merge same-speaker turns split by short pauses."""
    out: list[Turn] = []
    for t in sorted(raw, key=lambda t: (t.start, t.end)):
        start = max(t.start, out[-1].end) if out else t.start
        if t.end - start < min_len:
            continue
        if out and out[-1].speaker == t.speaker and start - out[-1].end <= merge_gap and t.end - out[-1].start <= max_len:
            out[-1].end = t.end
        else:
            out.append(Turn(start, t.end, t.speaker))
    return out


class StreamingSession:
    """Incremental diarization of one call. push() audio as it arrives, finish() at the end."""

    def __init__(self, diarizer: "NemotronDiarizer"):
        self.d = diarizer
        self.p = diarizer.processor
        self.audio = np.zeros(0, dtype=np.float32)
        self.offset = 0  # absolute sample index of self.audio[0]
        self.first = True
        self.mel_idx = 0
        self.start_idx = 0
        self.cache = None
        self.logits = []
        self.finished = False

    def _run(self, chunk: np.ndarray, **flags):
        import torch

        inputs = self.p(chunk, sampling_rate=TARGET_SR, is_streaming=True, **flags)
        inputs = inputs.to(self.d.model.device, dtype=self.d.model.dtype)
        with torch.inference_mode():
            out = self.d.model(**inputs, speaker_cache=self.cache)
        self.logits.append(out.logits.float().cpu())
        self.cache = out.speaker_cache

    def push(self, samples: np.ndarray) -> list[Turn]:
        with self.d.lock:
            self.audio = np.concatenate([self.audio, samples.astype(np.float32)])
            total = self.offset + len(self.audio)
            if self.first:
                if total < self.p.num_samples_first_audio_chunk:
                    return self.turns()
                self._run(self.audio[: self.p.num_samples_first_audio_chunk], is_first_audio_chunk=True)
                self.first = False
                self.mel_idx = self.p.num_mel_frames_per_step
                self.start_idx = self.p.audio_chunk_start(self.mel_idx)
            while self.start_idx + self.p.num_samples_per_audio_chunk <= total:
                a = self.start_idx - self.offset
                self._run(self.audio[a: a + self.p.num_samples_per_audio_chunk], is_first_audio_chunk=False)
                self.mel_idx += self.p.num_mel_frames_per_step
                self.start_idx = self.p.audio_chunk_start(self.mel_idx)
            drop = self.start_idx - self.offset  # audio before the next chunk start is never needed again
            if drop > 0:
                self.audio, self.offset = self.audio[drop:], self.offset + drop
            return self.turns()

    def finish(self) -> list[Turn]:
        with self.d.lock:
            if not self.finished:
                self.finished = True
                if self.first:  # shorter than one streaming chunk: score it in one go
                    if len(self.audio):
                        self._run(self.audio, is_first_audio_chunk=True, is_last_audio_chunk=True)
                else:
                    rest = self.audio[self.start_idx - self.offset:]
                    if len(rest):
                        self._run(rest, is_first_audio_chunk=False, is_last_audio_chunk=True)
            return self.turns()

    def turns(self) -> list[Turn]:
        if not self.logits:
            return []
        import torch

        logits = torch.cat(self.logits, dim=1)
        raw = self.p.extract_speaker_dict(logits)[0]
        return clean_turns([Turn(float(s["Start"]), float(s["End"]), speaker_label(int(s["Speaker"]))) for s in raw])


class NemotronDiarizer:
    name = "nemotron"

    def __init__(self, model_id: str, device: str = "auto", streaming_mode: str = "low_latency"):
        self.model_id, self.device_pref, self.streaming_mode = model_id, device, streaming_mode
        self.model = self.processor = None
        self.lock = threading.RLock()

    def load(self):
        with self.lock:
            if self.model is not None:
                return
            import torch
            from transformers import AutoModelForAudioFrameClassification, AutoProcessor

            device = self.device_pref if self.device_pref != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
            dtype = gpu_dtype(device)
            log.info("Loading %s on %s (%s)", self.model_id, device, dtype)
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.processor.set_streaming_mode(self.streaming_mode)
            self.model = AutoModelForAudioFrameClassification.from_pretrained(self.model_id, dtype=dtype).to(device).eval()

    def session(self) -> StreamingSession:
        self.load()
        return StreamingSession(self)

    def diarize(self, samples: np.ndarray) -> list[Turn]:
        s = self.session()
        s.push(samples)
        return s.finish()

    def info(self):
        info = {"backend": self.name, "model": self.model_id, "loaded": self.model is not None}
        if self.model is not None:
            info["device"] = str(self.model.device)
        return info


class FakeSession:
    def __init__(self, diarizer):
        self.d, self.audio = diarizer, np.zeros(0, dtype=np.float32)

    def push(self, samples):
        self.audio = np.concatenate([self.audio, samples.astype(np.float32)])
        return self.d.diarize(self.audio)

    def finish(self):
        return self.d.diarize(self.audio)

    def turns(self):
        return self.d.diarize(self.audio)


class FakeDiarizer:
    """Tests: every pause-separated utterance alternates speaker 1, speaker 2, speaker 1..."""

    name = "fake"

    def diarize(self, samples):
        return [Turn(a / TARGET_SR, b / TARGET_SR, speaker_label(i % 2)) for i, (a, b) in enumerate(utterances(samples))]

    def session(self):
        return FakeSession(self)

    def load(self):
        pass

    def info(self):
        return {"backend": self.name}


_diarizer = None
_lock = threading.Lock()
_sessions: dict[int, object] = {}  # live calls: call id -> open session (one gunicorn worker, so in-process is fine)


def get_diarizer():
    """The configured diarizer, or None when DIARIZER=off."""
    global _diarizer
    with _lock:
        if _diarizer is None and settings.DIARIZER != "off":
            if settings.DIARIZER == "fake":
                _diarizer = FakeDiarizer()
            elif settings.DIARIZER == "nemotron":
                _diarizer = NemotronDiarizer(settings.DIARIZER_MODEL_ID, settings.ASR_DEVICE, settings.DIARIZER_STREAMING_MODE)
            elif settings.DIARIZER == "demo":
                from .demo import DemoDiarizer
                _diarizer = DemoDiarizer()
            else:
                raise ValueError(f"Unknown DIARIZER {settings.DIARIZER!r} (use 'nemotron', 'demo', 'fake' or 'off')")
        return _diarizer


def set_diarizer(diarizer):
    global _diarizer
    with _lock:
        _diarizer = diarizer
        _sessions.clear()


def live_session(call_id: int, create: bool = True):
    d = get_diarizer()
    if d is None:
        return None
    with _lock:
        if call_id not in _sessions and create:
            _sessions[call_id] = d.session()
        return _sessions.get(call_id)


def close_live_session(call_id: int):
    with _lock:
        _sessions.pop(call_id, None)
