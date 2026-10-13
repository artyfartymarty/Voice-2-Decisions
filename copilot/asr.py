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

from .audio import TARGET_SR, utterances

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
        for start, end in utterances(samples, max_seconds=settings.ASR_CHUNK_SECONDS):
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


class NemoSpeechASR(ASRBackend):
    """Client for the speech service (NVIDIA NeMo-Speech.cpp `nemo-speech serve`, see asr/Dockerfile).

    English Nemotron Q8 GGUF, ~4x faster than the Transformers path on the same GPU, with word boost
    (`speech_contexts`) from the reference data. With diarize=True the service also tags each word with a speaker
    (Nemotron-3-Diarization), so a mono recording gets speakers in the same request.
    """

    name = "nemo_speech"
    supports_speakers = True

    def __init__(self, url: str, api_key: str = "", boost: float = 4.0, timeout: float = 300.0, pause_s: float = 0.6):
        self.url = url.rstrip("/")
        self.api_key, self.boost, self.timeout, self.pause_s = api_key, boost, timeout, pause_s

    def _request(self, samples: np.ndarray, diarize: bool) -> dict:
        import io
        import json
        import urllib.request
        import uuid

        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, samples, TARGET_SR, format="WAV", subtype="PCM_16")
        fields = {"response_format": "verbose_json", "diarization": "true" if diarize else "false"}
        phrases = _boost_phrases()
        if phrases and self.boost:
            fields["speech_contexts"] = json.dumps([{"phrases": phrases, "boost": self.boost}])
        boundary = uuid.uuid4().hex
        body = b"".join(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
                        for k, v in fields.items())
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n"
                 f"Content-Type: audio/wav\r\n\r\n").encode() + buf.getvalue() + f"\r\n--{boundary}--\r\n".encode()
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.url}/v1/audio/transcriptions", data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def _segments(self, words: list[dict], by_speaker: bool) -> list[tuple[Segment, str]]:
        """Group words into segments at pauses (and, with speakers, at every change of speaker)."""
        out: list[tuple[Segment, str]] = []
        for w in words:
            spk = f"speaker {w['speaker']}" if by_speaker and w.get("speaker") is not None else ""
            text = w.get("word", "").strip()
            if not text:
                continue
            if out and out[-1][1] == spk and float(w["start"]) - out[-1][0].end < self.pause_s:
                seg = out[-1][0]
                seg.end, seg.text = float(w["end"]), f"{seg.text} {text}"
            else:
                out.append((Segment(float(w["start"]), float(w["end"]), text), spk))
        return out

    def transcribe(self, samples: np.ndarray) -> Transcript:
        t = Transcript(duration=len(samples) / TARGET_SR, language="en-US")
        if len(samples) < TARGET_SR // 10:
            return t
        data = self._request(samples, diarize=False)
        if data.get("words"):
            t.segments = [seg for seg, _ in self._segments(data["words"], by_speaker=False)]
        elif data.get("text"):
            t.segments = [Segment(0.0, t.duration, data["text"].strip())]
        return t

    def transcribe_with_speakers(self, samples: np.ndarray) -> list[tuple[Segment, str]]:
        """Mono recording -> [(segment, "speaker N")], one request for ASR + diarization."""
        if len(samples) < TARGET_SR // 10:
            return []
        return self._segments(self._request(samples, diarize=True).get("words", []), by_speaker=True)

    def info(self):
        info = {"backend": self.name, "url": self.url, "model": "nemotron-speech-streaming-en-0.6b Q8 (NeMo-Speech.cpp)"}
        try:
            import json
            import urllib.request

            with urllib.request.urlopen(f"{self.url}/ready", timeout=2) as r:
                info["ready"] = json.loads(r.read())
        except Exception as exc:
            info["ready"] = {"ready": False, "error": str(exc)}
        return info


def _boost_phrases() -> list[str]:
    try:
        from . import extract
        return extract.get().boost_words()
    except Exception:
        return []


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
            elif settings.ASR_BACKEND == "nemo_speech":
                _backend = NemoSpeechASR(settings.NEMO_SPEECH_URL, settings.NEMO_SPEECH_API_KEY, settings.ASR_WORD_BOOST)
            else:
                raise ValueError(f"Unknown ASR_BACKEND {settings.ASR_BACKEND!r} (use 'nemo_speech', 'nemotron' or 'fake')")
        return _backend


def set_backend(backend: ASRBackend | None):
    """Swap the process-wide backend (tests use this)."""
    global _backend
    with _backend_lock:
        _backend = backend
