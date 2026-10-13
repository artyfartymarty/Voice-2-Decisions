"""The decision model: which function's result answers what the caller just asked for?

WinnowDecider asks Winnow-12B one typed `choice` question on winnow-server (POST /v1/systemone, EldanRing/
winnow-inference). The options are the functions (by their descriptions) plus "none"; the state is the caller's latest
line, the last few lines of the call and the variables filled so far. Winnow returns a probability per option, so the
pick comes with a confidence and nothing is generated.

RuleDecider is a keyword fallback: used when WINNOW_URL is empty, and for 30 s whenever Winnow can't be reached.
Both return {"function": name or None, "p": float, "by": "winnow" | "rules", "probabilities": {...}}.
"""
import json
import logging
import re
import time
import urllib.request

from .functions import FUNCTIONS, NONE_DESCRIPTION

log = logging.getLogger(__name__)
_I = re.IGNORECASE
RULES = [  # first match wins, so the more specific asks come first
    ("escalate", re.compile(r"\b(manager|supervisor|unacceptable|take (our|my) business|fed up|sick of|ridiculous|furious)\b", _I)),
    ("dispute_check", re.compile(r"\b(dispute|charged|overcharg\w*|invoice|bill(ed)?|surprise|weren'?t expecting|what (will )?the fees|which fees|what fees)\b", _I)),
    ("fuel_quote", re.compile(r"\b(quotes?|price|pricing|per gallon|how much|rates?|cost)\b", _I)),
    ("location_info", re.compile(r"\b(hours|open|close[sd]?|notice|ppr|fbo)\b", _I)),
    ("tail_check", re.compile(r"\b(tail|registration)\b", _I)),
    ("customer_profile", re.compile(r"\b(on file|my account|our account|what do you have)\b", _I)),
]


UNSURE_WHAT = {"customer_number": "customer", "icao": "airport or FBO"}


class RuleDecider:
    name = "rules"

    def decide(self, latest: str, recent: list[dict], variables: dict, unsure: dict | None = None) -> dict:
        # No model to ask "was a tail / a customer name said?": try the extract model when the line is about a tail,
        # or when the list matcher saw something like a customer name in it.
        unsure = unsure or {}
        said = {}
        if unsure.get("tail_said"):
            said["tail_said"] = 1.0
        if unsure.get("customer_said"):
            said["customer_said"] = 1.0 if unsure["customer_said"].get("hint") else 0.0
        extra = {"variables": said} if said else {}
        for fn, pattern in RULES:
            if pattern.search(latest):
                return {"function": fn, "p": 1.0, "by": self.name, **extra}
        return {"function": None, "p": 1.0, "by": self.name, **extra}

    def choose(self, what: str, heard: str, latest: str, recent: list[dict], options: dict) -> dict | None:
        return None  # no model to ask: the caller keeps the old fuzzy behaviour


class WinnowDecider:
    name = "winnow"

    def __init__(self, url: str, api_key: str = "", timeout: float = 3.0, model: str = "Winnow-12B"):
        self.url = url.rstrip("/") + "/v1/systemone"
        self.api_key, self.timeout, self.model = api_key, timeout, model
        self.fallback = RuleDecider()
        self._down_until = 0.0

    def request_body(self, latest: str, recent: list[dict], variables: dict, unsure: dict | None = None) -> dict:
        criteria = {f.name: f.description for f in FUNCTIONS}
        criteria["none"] = NONE_DESCRIPTION
        return {
            "model": self.model,
            "state": {
                "call": "A caller is on the phone with an aviation fuel desk agent.",
                "recent_lines": [f"{line['speaker']}: {line['text']}" for line in recent[-6:]],
                "caller_latest_line": latest,
                "known_so_far": {k: v["label"] or v["value"] for k, v in variables.items()},
            },
            "questions": {
                "answer_with": {
                    "type": "choice",
                    "instructions": "Considering the caller's latest line in context, what is the caller asking for "
                                    "right now? Pick the lookup whose result the agent should give them.",
                    "criteria": criteria,
                },
                # Tie-breakers for fuzzy matches on this line; they share the state, so they cost a few tokens each.
                **{f"which_{var}": {
                    "type": "choice",
                    "instructions": f"The caller's speech was transcribed as \"{u['heard']}\". Which "
                                    f"{UNSURE_WHAT.get(var, var)} did they mean?",
                    "criteria": {**u["options"], "none": "none of these"},
                } for var, u in (unsure or {}).items() if isinstance(u, dict) and u.get("options")},
                # Asked only when the extract model is configured (EXTRACT_MODEL_URL): on yes it pulls the value out.
                **({"tail_said": {
                    "type": "noul",
                    "instructions": "In their latest line, did the caller say a complete aircraft tail number "
                                    "(registration), all of its letters and digits?",
                }} if (unsure or {}).get("tail_said") else {}),
                **({"customer_said": {
                    "type": "noul",
                    "instructions": "In their latest line, did the caller say the name of the company or customer "
                                    "account they are calling from or about?",
                }} if (unsure or {}).get("customer_said") else {}),
            },
        }

    def decide(self, latest: str, recent: list[dict], variables: dict, unsure: dict | None = None) -> dict:
        if time.monotonic() < self._down_until:
            return {**self.fallback.decide(latest, recent, variables), "by": "rules (winnow unavailable)"}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = json.dumps(self.request_body(latest, recent, variables, unsure)).encode()
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                answers = json.loads(resp.read())["answers"]
            answer = answers["answer_with"]
            probs = {k: float(v) for k, v in (answer.get("probabilities") or answer.get("probs") or {}).items()}
            pick = answer.get("choice") or answer.get("selected") or answer.get("answer") or (max(probs, key=probs.get) if probs else None)
            if pick is None:
                raise ValueError(f"unrecognised choice answer: {answer}")
        except Exception as exc:
            log.warning("winnow decision unavailable (%s); using keyword rules for 30 s", exc)
            self._down_until = time.monotonic() + 30
            return {**self.fallback.decide(latest, recent, variables), "by": "rules (winnow unavailable)"}
        settled = {}
        for key, a in answers.items():
            if key.startswith("which_") and a.get("choice") not in (None, "none"):
                settled[key[len("which_"):]] = {"value": a["choice"], "p": round(float((a.get("probabilities") or {}).get(a["choice"], 0)), 3)}
        for flag in ("tail_said", "customer_said"):
            if flag in answers:
                settled[flag] = float(answers[flag].get("noul", 0.0))
        return {"function": None if pick == "none" else pick, "p": round(probs.get(pick, answer.get("confidence", 0.0)), 3),
                "confidence": answer.get("confidence"), "probabilities": probs, "by": self.name, "variables": settled,
                "ms": round((time.perf_counter() - t0) * 1000)}


    def choose(self, what: str, heard: str, latest: str, recent: list[dict], options: dict) -> dict | None:
        """A name or tail with no exact match: which of these close ones did the caller mean, or none of them?
        Returns {"value": option or None for none, "p": float}, or None when Winnow can't be reached."""
        if time.monotonic() < self._down_until:
            return None
        body = {"model": self.model,
                "state": {"call": "A caller is on the phone with an aviation fuel desk agent.",
                          "recent_lines": [f"{line['speaker']}: {line['text']}" for line in recent[-6:]],
                          "caller_latest_line": latest},
                "questions": {"which": {
                    "type": "choice",
                    "instructions": f"Speech-to-text heard the {what} as \"{heard}\", which isn't on file exactly. It "
                                    f"may be misheard. Which of these {what}s on file did the caller mean? Pick none "
                                    f"if none of them fits.",
                    "criteria": {**options, "none": f"none of these; a {what} that isn't on file"}}}}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                a = json.loads(resp.read())["answers"]["which"]
            probs = {k: float(v) for k, v in (a.get("probabilities") or {}).items()}
            pick = a.get("choice") or (max(probs, key=probs.get) if probs else None)
        except Exception as exc:
            log.warning("winnow choice unavailable (%s)", exc)
            return None
        if pick is None:
            return None
        return {"value": None if pick == "none" else pick, "p": round(probs.get(pick, a.get("confidence") or 0.0), 3)}


def from_env(env) -> RuleDecider | WinnowDecider:
    url = env.get("WINNOW_URL", "")
    if env.get("DECIDER", "auto") == "rules" or not url:
        return RuleDecider()
    return WinnowDecider(url, env.get("WINNOW_API_KEY", ""), float(env.get("WINNOW_TIMEOUT", "3")))
