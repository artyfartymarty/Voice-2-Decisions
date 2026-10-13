"""Sentiment for the Customer Experience use case: the one path that uses a model.

WinnowSentiment asks a running winnow-server (EldanRing/winnow-inference, POST /v1/systemone) a typed `score`
question about the speaker's latest line, in the context of their recent lines. LexiconSentiment is a word-list
fallback used by tests, CPU-only boxes, and whenever the Winnow server can't be reached.

Both return a score in [-1, 1] (negative = frustrated) plus where it came from.
"""
import json
import logging
import re
import time
import urllib.request

log = logging.getLogger(__name__)

_I = re.IGNORECASE
LEXICON = [
    (re.compile(r"\b(frustrat\w*|annoy\w*|angry|furious|upset|unacceptable|ridiculous|terrible|awful|horrible|disappoint\w*"
                r"|overcharg\w*|ripped off|rip-?off|waste of|worst|complain\w*|fed up|sick of|never again|not happy|unhappy|hate)\b", _I), -1.0),
    (re.compile(r"\b(problem|wrong|late|delay(ed)?|too expensive|confus\w*|mistake)\b", _I), -0.5),
    (re.compile(r"\b(great|perfect|excellent|appreciate\w*|thank(s| you)|wonderful|happy|love|awesome|fantastic|sounds good|glad|helpful)\b", _I), 1.0),
]
MOOD_SCALE = ["positive or pleased", "neutral or matter-of-fact", "mildly frustrated or impatient", "frustrated or angry"]


def lexicon_score(text: str) -> float:
    total = sum(weight * len(pat.findall(text)) for pat, weight in LEXICON)
    return max(-1.0, min(1.0, total))


class LexiconSentiment:
    name = "lexicon"

    def score(self, speaker: str, latest: str, recent: list[str]) -> dict:
        return {"score": lexicon_score(latest), "backend": self.name}


class WinnowSentiment:
    """Typed decision on Winnow: 'How does the {speaker} sound in their latest line?' on a 4-step scale."""

    name = "winnow"

    def __init__(self, url: str, api_key: str = "", timeout: float = 2.0, model: str = "Winnow-12B"):
        self.url = url.rstrip("/") + "/v1/systemone"
        self.api_key, self.timeout, self.model = api_key, timeout, model
        self.fallback = LexiconSentiment()
        self._down_until = 0.0

    def request_body(self, speaker: str, latest: str, recent: list[str]) -> dict:
        who = speaker or "speaker"
        return {
            "model": self.model,
            "state": {"speaker": who, "earlier_lines_from_this_speaker": recent[-4:], "latest_line": latest},
            "questions": {
                "mood": {"type": "score",
                         "instructions": f"On a customer service call, how does the {who} sound in their latest line?",
                         "criteria": MOOD_SCALE},
            },
        }

    def score(self, speaker: str, latest: str, recent: list[str]) -> dict:
        if time.monotonic() < self._down_until:
            return {**self.fallback.score(speaker, latest, recent), "backend": "lexicon (winnow unavailable)"}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url, data=json.dumps(self.request_body(speaker, latest, recent)).encode(),
                                     headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                answer = json.loads(resp.read())["answers"]["mood"]
            expected_value(answer)  # validate before trusting it
        except Exception as exc:
            log.warning("winnow sentiment unavailable (%s); using the lexicon for 30 s", exc)
            self._down_until = time.monotonic() + 30
            return {**self.fallback.score(speaker, latest, recent), "backend": "lexicon (winnow unavailable)"}
        expected, probs = expected_value(answer)
        score = round((1.0 - expected) / (1.0 if expected <= 1.0 else 2.0), 3)  # 0 -> +1, 1 -> 0, 3 -> -1
        return {"score": max(-1.0, min(1.0, score)), "backend": self.name,
                "confidence": answer.get("confidence"), "probabilities": probs}


def expected_value(answer: dict) -> tuple[float, dict]:
    """Expected index on the MOOD_SCALE from a /v1/systemone score answer.

    The API returns a probability map keyed by zero-based indices plus an expected value; field names are read
    defensively (score / expected / expected_value / value), and computed from the probabilities if absent.
    """
    probs = next((v for v in answer.values() if isinstance(v, dict) and v and all(str(k).isdigit() for k in v)), {})
    for key in ("score", "expected", "expected_value", "value"):  # Winnow a50c7d0 returns it as "score"
        if isinstance(answer.get(key), (int, float)):
            return float(answer[key]), probs
    if probs:
        return sum(int(k) * float(p) for k, p in probs.items()) / max(sum(map(float, probs.values())), 1e-9), probs
    raise ValueError(f"unrecognised score answer: {answer}")


_scorer = None


def get_scorer():
    global _scorer
    if _scorer is None:
        from django.conf import settings

        backend = settings.SENTIMENT_BACKEND
        if backend == "auto":
            backend = "winnow" if settings.WINNOW_URL else "lexicon"
        _scorer = WinnowSentiment(settings.WINNOW_URL, settings.WINNOW_API_KEY) if backend == "winnow" else LexiconSentiment()
    return _scorer


def set_scorer(scorer):
    global _scorer
    _scorer = scorer
