"""One call: transcript lines in; variables, auto-called functions and the decision out.

    line ──► variables ──► functions whose `needs` are all filled are called ──► results wait in the call
      │                                                                              │
      └─(caller, final)─► decision model: what is the caller asking for? ───────────┴─► surface that result

If the caller asks for something whose variables aren't filled yet, the decision says what to ask for, and the result
is surfaced on its own as soon as the function fires (no second model call).

Variables come from the deterministic extractor the rest of the repo uses (copilot/extract.py, copilot/slots.py), so
only values that exist in the reference data, or a valid registration, can fill one:

    customer_number   account number or company name, fuzzy + phonetic ("North Wind Aviation" -> 204511)
    icao              ICAO / IATA, spelled or phonetic, FBO or airport name ("Van Nuys" -> KVNY)
    tail_number       registration over several breaths, re-asks and corrections ("November eight eight ..." -> N88NW)
"""
import re
import time

from copilot import extract, slots

from .functions import BY_NAME, FUNCTIONS

SENTENCE = re.compile(r"(?<=[.!?])\s+")


class Call:
    def __init__(self, call_id: str, extractor, decider, min_p: float = 0.5, decide_on=("caller",)):
        self.id = call_id
        self.extractor, self.decider = extractor, decider
        self.min_p, self.decide_on = min_p, set(decide_on)
        self.variables: dict[str, dict] = {}  # name -> {value, label, confidence, heard, speaker}
        self.tail_slot = slots.new_slot()
        self.results: dict[str, dict] = {}  # function -> {args, result, at}
        self._fired_with: dict[str, tuple] = {}
        self.transcript: list[dict] = []
        self.preview = ""  # latest SPECULATIVE text, display only
        self.waiting_for: str | None = None  # function the caller asked for whose variables aren't filled yet
        self.decision: dict | None = None
        self.prompts: list[str] = []
        self.started = time.time()

    # --- variables ---
    def _set(self, name, value, label="", confidence=1.0, heard="", speaker=""):
        if self.variables.get(name, {}).get("value") == value:
            return None
        self.variables[name] = {"value": value, "label": label, "confidence": confidence, "heard": heard, "speaker": speaker}
        return name

    def _extract(self, speaker: str, text: str) -> tuple[list[str], list[str]]:
        changed, prompts = [], []
        for sentence in (s for s in SENTENCE.split(text) if s.strip()):
            found = self.extractor.extract(sentence)
            if "customer" in found:
                e = found["customer"]
                changed.append(self._set("customer_number", e.value, e.label, e.confidence, e.evidence, speaker))
            if "location" in found:
                e = found["location"]
                changed.append(self._set("icao", e.value, e.label, e.confidence, e.evidence, speaker))
            customer = self.variables.get("customer_number", {}).get("value")
            for ev in slots.update(self.tail_slot, sentence, speaker, self.extractor, customer):
                if ev["type"] == "validate":
                    changed.append(self._set("tail_number", ev["tail_number"], ev["country"], ev["confidence"], ev["heard"], speaker))
                else:
                    prompts.append(ev["message"])
        return [c for c in changed if c], prompts

    # --- functions ---
    def _args(self, fn) -> dict | None:
        if any(n not in self.variables for n in fn.needs):
            return None
        return {n: self.variables[n]["value"] for n in fn.needs + fn.optional if n in self.variables}

    def call(self, name: str) -> dict:
        fn = BY_NAME[name]
        args = self._args(fn) or {}
        result = fn.run(self.extractor.ref, self.extractor.fleet, **args)
        self.results[name] = {"args": args, "result": result, "at": round(time.time() - self.started, 2)}
        self._fired_with[name] = tuple(sorted(args.items()))
        return {"function": name, **self.results[name]}

    def _autocall(self) -> list[dict]:
        fired = []
        for fn in FUNCTIONS:
            args = self._args(fn)
            if fn.auto and args is not None and self._fired_with.get(fn.name) != tuple(sorted(args.items())):
                fired.append(self.call(fn.name))
        return fired

    # --- decision ---
    def missing(self, name: str) -> list[str]:
        return [n for n in BY_NAME[name].needs if n not in self.variables]

    def _surface(self, name: str, picked: dict, line: str) -> dict:
        missing = self.missing(name)
        fn = BY_NAME[name]
        if missing:
            self.waiting_for = name
            ask = " and ".join(fn.asks.get(m, m) for m in missing)
            return {"function": name, "ready": False, "missing": missing, "say": f"Ask the caller for {ask}.",
                    "because": line, **picked}
        self.waiting_for = None
        if name not in self.results or not fn.auto:
            self.call(name)
        return {"function": name, "ready": True, "result": self.results[name]["result"],
                "say": self.results[name]["result"].get("say", ""), "because": line, **picked}

    def feed(self, speaker: str, text: str, final: bool = True) -> dict:
        if not final:
            self.preview = f"{speaker}: {text}"
            return {"final": False, "preview": self.preview}
        self.preview = ""
        line = {"speaker": speaker, "text": text.strip(), "at": round(time.time() - self.started, 2)}
        self.transcript.append(line)
        changed, prompts = self._extract(speaker, line["text"])
        self.prompts += prompts
        fired = self._autocall()
        decision = None
        if speaker in self.decide_on:
            picked = self.decider.decide(line["text"], self.transcript, self.variables)
            picked = {k: picked[k] for k in ("p", "by", "probabilities") if k in picked} | {"function": picked["function"]}
            if picked["function"] and picked["p"] >= self.min_p:
                decision = self._surface(picked.pop("function"), picked, line["text"])
        if decision is None and self.waiting_for and not self.missing(self.waiting_for):
            # The caller asked earlier; the last variable just arrived, so the answer is ready now.
            decision = self._surface(self.waiting_for, {"by": "waiting"}, line["text"])
        if decision:
            self.decision = decision
        return {"final": True, "line": line, "variables_changed": changed, "fired": fired, "prompts": prompts,
                "decision": decision}

    def state(self) -> dict:
        return {"call_id": self.id, "variables": self.variables, "results": self.results, "decision": self.decision,
                "waiting_for": self.waiting_for, "prompts": self.prompts, "transcript": self.transcript,
                "preview": self.preview}


def load_extractor(reference_dir: str, tail_min_confidence: float = 0.8):
    return extract._extractor(str(reference_dir), tail_min_confidence)
