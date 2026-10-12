"""Speech-to-text backends.

NemotronASR runs nvidia/nemotron-3.5-asr-streaming-0.6b through Hugging Face Transformers.
FakeASR returns fixed text so tests and CPU-only machines can run the whole app without a model.
"""
import logging
import re
import threading
from dataclasses import dataclass, field

import numpy as np
from django.conf import settings

from .audio import TARGET_SR, split_on_silence

log = logging.getLogger(__name__)
LANG_TAG = re.compile(r"<([a-z]{2,3}-[A-Z]{2})>")


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    segments: list[Segment] = field(default_factory=list)
    language: str = ""
    duration: float = 0.0

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments if s.text).strip()


class ASRBackend:
    name = "base"

    def transcribe_chunk(self, samples: np.ndarray) -> tuple[str, str]:
        """Return (text, detected_language) for one chunk of 16 kHz mono audio."""
        raise NotImplementedError

    def transcribe(self, samples: np.ndarray) -> Transcript:
        transcript = Transcript(duration=len(samples) / TARGET_SR)
        for start, end in split_on_silence(samples, max_seconds=settings.ASR_CHUNK_SECONDS):
            text, lang = self.transcribe_chunk(samples[start:end])
            if lang and not transcript.language:
                transcript.language = lang
            if text:
                transcript.segments.append(Segment(start / TARGET_SR, end / TARGET_SR, text))
        return transcript

    def info(self) -> dict:
        return {"backend": self.name}


class FakeASR(ASRBackend):
    """Deterministic stand-in. Returns FAKE_ASR_TEXT (or a canned call) once per transcribe()."""

    name = "fake"
    DEFAULT_TEXT = (
        "Thanks for joining. The budget is $40,000 for the pilot. "
        "We decided to go with the annual plan. "
        "Sarah will send the contract by Friday. "
        "My concern is the onboarding timeline. "
        "Can you confirm the start date?"
    )

    def __init__(self, text: str | None = None):
        self.text = text

    def transcribe_chunk(self, samples):
        return (self.text if self.text is not None else (settings.FAKE_ASR_TEXT or self.DEFAULT_TEXT)), "en-US"

    def transcribe(self, samples):
        text, lang = self.transcribe_chunk(samples)
        duration = len(samples) / TARGET_SR
        return Transcript(segments=[Segment(0.0, duration, text)] if text else [], language=lang, duration=duration)


class NemotronASR(ASRBackend):
    name = "nemotron"

    def __init__(self, model_id: str, language: str = "en-US", device: str = "auto"):
        self.model_id = model_id
        self.language = language
        self.device_pref = device
        self._model = None
        self._processor = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForRNNT, AutoProcessor

        device = self.device_pref
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        log.info("Loading %s on %s (%s)", self.model_id, device, dtype)
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForRNNT.from_pretrained(self.model_id, dtype=dtype).to(device).eval()

    def load(self):
        with self._lock:
            self._load()

    def transcribe_chunk(self, samples):
        import torch

        if len(samples) < TARGET_SR // 10:  # under 100 ms: nothing to hear
            return "", ""
        with self._lock:
            self._load()
            inputs = self._processor(samples, sampling_rate=TARGET_SR, language=self.language, return_tensors="pt")
            inputs = inputs.to(self._model.device, dtype=self._model.dtype)
            with torch.inference_mode():
                out = self._model.generate(**inputs, return_dict_in_generate=True)
            raw = self._processor.decode(out.sequences, skip_special_tokens=False)
            text = self._processor.decode(out.sequences, skip_special_tokens=True)
        raw = raw[0] if isinstance(raw, list) else raw
        text = text[0] if isinstance(text, list) else text
        match = LANG_TAG.search(raw)
        lang = match.group(1) if match else (self.language if self.language != "auto" else "")
        return LANG_TAG.sub("", text).strip(), lang

    def info(self):
        info = {"backend": self.name, "model": self.model_id, "language": self.language, "loaded": self._model is not None}
        if self._model is not None:
            info["device"] = str(self._model.device)
        return info


_backend: ASRBackend | None = None
_backend_lock = threading.Lock()


def get_backend() -> ASRBackend:
    global _backend
    with _backend_lock:
        if _backend is None:
            if settings.ASR_BACKEND == "fake":
                _backend = FakeASR()
            elif settings.ASR_BACKEND == "nemotron":
                _backend = NemotronASR(settings.ASR_MODEL_ID, settings.ASR_LANGUAGE, settings.ASR_DEVICE)
            else:
                raise ValueError(f"Unknown ASR_BACKEND {settings.ASR_BACKEND!r} (use 'nemotron' or 'fake')")
        return _backend


def set_backend(backend: ASRBackend | None):
    """Swap the process-wide backend (tests use this)."""
    global _backend
    with _backend_lock:
        _backend = backend
