"""Guardrails against the easy mistakes: a server quietly running on the fake speech model, a typo in a backend
name that only surfaces on the first request, and transcripts that are obviously wrong (the same text for every
chunk, or nothing from audio that has speech in it).

- config_problems() lists what is wrong with the settings. The `v2d.E001` system check (runserver, migrate,
  `manage.py check`) and v2d/wsgi.py (gunicorn) refuse to start on any of them.
- TranscriptCheck watches a call's transcript and records warnings in call.context["warnings"], which the API,
  the summary and the logs all show.
"""
import logging
import re

from django.conf import settings
from django.core import checks
from django.core.exceptions import ImproperlyConfigured

log = logging.getLogger(__name__)

CHOICES = {
    "ASR_BACKEND": ("nemo_speech", "nemotron", "fake"),
    "DIARIZER": ("nemotron", "fake", "off"),
    "LEG_MODE": ("separate", "mix"),
    "SENTIMENT_BACKEND": ("auto", "winnow", "lexicon"),
    "DIARIZER_STREAMING_MODE": ("low_latency", "very_low_latency", "ultra_low_latency"),
}
FAKE_HINT = ("it returns canned text, so every transcript is the same paragraph. Set ALLOW_FAKE_BACKEND=1 only for "
             "a CPU-only demo or UI work; otherwise set ASR_BACKEND=nemo_speech (or nemotron) and DIARIZER=nemotron.")


def fake_backends() -> list[str]:
    """Settings that select a fake backend."""
    return [name for name in ("ASR_BACKEND", "DIARIZER") if getattr(settings, name, "") == "fake"]


def fake_in_use() -> bool:
    """True when the speech model actually serving requests is the fake one."""
    from .asr import get_backend
    try:
        return get_backend().name == "fake"
    except ValueError:  # a misspelled ASR_BACKEND; config_problems() reports it
        return False


def config_problems() -> list[str]:
    """Settings that would make the server run wrong. Empty when it is safe to serve."""
    problems = [f"{name}={getattr(settings, name)!r} is not one of {', '.join(allowed)}"
                for name, allowed in CHOICES.items() if getattr(settings, name, allowed[0]) not in allowed]
    fakes = fake_backends()
    if fakes and not getattr(settings, "ALLOW_FAKE_BACKEND", False):
        problems.append(f"{' and '.join(f'{n}=fake' for n in fakes)}: {FAKE_HINT}")
    return problems


def enforce():
    """Refuse to serve with a broken configuration; warn loudly when the fake backend is allowed on purpose."""
    problems = config_problems()
    if problems:
        raise ImproperlyConfigured("Voice-2-Decisions will not start:\n  - " + "\n  - ".join(problems))
    if fake_backends():
        log.warning("FAKE SPEECH BACKEND (%s): transcripts are canned text, not what was said.", ", ".join(fake_backends()))


@checks.register()
def check_settings(app_configs=None, **kwargs):
    return [checks.Error(p, id="v2d.E001") for p in config_problems()]


# --- transcript sanity ---

def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


class TranscriptCheck:
    """Flags transcripts that can't be right. State lives in call.context so it survives between live chunks.

    repeated_text   the same text (of MIN_WORDS or more) came back REPEAT_LIMIT times in a row (a stuck or fake model)
    no_text         audio with speech in it came back empty EMPTY_LIMIT times in a row (ASR down, wrong rate, muted)
    fake_backend    the fake speech model produced this transcript
    """

    REPEAT_LIMIT = 3
    MIN_WORDS = 4  # shorter texts don't count as repeats
    EMPTY_LIMIT = 3

    def __init__(self, call):
        self.call = call
        self.state = call.context.setdefault("transcript_check", {"last": "", "repeats": 0, "empty": 0})

    def heard(self, text: str, had_speech: bool = True):
        """One transcription result (a live chunk's span, or a recording's turn)."""
        norm = _norm(text)
        if not norm:
            self.state["empty"] = self.state["empty"] + 1 if had_speech else 0
            if self.state["empty"] >= self.EMPTY_LIMIT:
                self.warn("no_text", f"{self.state['empty']} pieces of audio with speech in them came back with no "
                                     "text. Check that the speech service is up and the audio is not silent.")
            return
        self.state["empty"] = 0
        if len(norm.split()) < self.MIN_WORDS:  # "yeah", "okay": short replies repeat on real calls
            return
        self.state["repeats"] = self.state["repeats"] + 1 if norm == self.state["last"] else 1
        self.state["last"] = norm
        if self.state["repeats"] >= self.REPEAT_LIMIT:
            self.warn("repeated_text", f"the same text came back {self.state['repeats']} times in a row: "
                                       f"\"{text[:80]}\". The speech model is stuck or fake.")
        if fake_in_use():
            self.warn("fake_backend", "this transcript comes from the fake speech model (ASR_BACKEND=fake): it is "
                                      "canned text, not what was said.")

    def warn(self, code: str, message: str):
        warnings = self.call.context.setdefault("warnings", [])
        if any(w["code"] == code for w in warnings):
            return
        warnings.append({"code": code, "message": message, "at": round(self.call.duration, 2)})
        log.warning("call %s: transcript check %s: %s", self.call.pk, code, message)
