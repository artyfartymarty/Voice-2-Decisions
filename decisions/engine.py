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
from difflib import SequenceMatcher

from copilot import extract, slots

from .functions import BY_NAME, FUNCTIONS

SENTENCE = re.compile(r"(?<=[.!?])\s+")


class Call:
    def __init__(self, call_id: str, extractor, decider, min_p: float = 0.5, decide_on=("caller",), normalizer=None):
        self.id = call_id
        self.extractor, self.decider, self.normalizer = extractor, decider, normalizer
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
        self._unsure: dict[str, dict] = {}  # this line's fuzzy matches, for Winnow to settle: var -> {heard, options}

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
                name = self._set("customer_number", e.value, e.label, e.confidence, e.evidence, speaker)
                changed.append(name)
                if name and e.confidence < 0.95:  # matched on a heard name, not the account number: read it back
                    prompts.append(f"Confirm the account: heard \"{e.evidence}\", matched {e.label} ({e.value}).")
                    self._unsure["customer_number"] = self._candidates(e, [(n, c, name_) for n, (name_, c) in self.extractor.customers.items()])
            if "location" in found:
                e = found["location"]
                if self._set("icao", e.value, e.label, e.confidence, e.evidence, speaker):
                    changed.append("icao")
                    if e.confidence < 0.95:
                        places = [(icao, alias, self.extractor.ref.locations[icao].fbo) for icao, alias in self.extractor.place_names]
                        self._unsure["icao"] = self._candidates(e, places)
            customer = self.variables.get("customer_number", {}).get("value")
            for ev in slots.update(self.tail_slot, sentence, speaker, self.extractor, customer):
                if ev["type"] == "validate":
                    changed.append(self._set("tail_number", ev["tail_number"], ev["country"], ev["confidence"], ev["heard"], speaker))
                else:
                    prompts.append(ev["message"])
        return [c for c in changed if c], prompts

    @staticmethod
    def _candidates(match, names: list[tuple[str, str, str]], top: int = 4) -> dict:
        """The fuzzy match plus the next-closest names to what was heard: the options Winnow chooses between."""
        best: dict[str, tuple[float, str]] = {match.value: (2.0, match.label)}
        for value, name, label in names:
            score = SequenceMatcher(None, extract.squash(match.evidence), extract.squash(name)).ratio()
            if score > best.get(value, (0.0, ""))[0]:
                best[value] = (score, label)
        ranked = sorted(best.items(), key=lambda kv: -kv[1][0])[:top]
        return {"heard": match.evidence, "options": {value: label for value, (score, label) in ranked if score >= 0.3}}

    def _normalize_tail(self, text: str, speaker: str, prompts: list[str]) -> list[str]:
        """Winnow heard a full tail the parser missed: ask the external normalizer, then validate what it returns."""
        tail = self.normalizer.normalize(text, [f"{l['speaker']}: {l['text']}" for l in self.transcript[-4:-1]])
        country = extract.registration_country(tail or "")
        if not country:
            return []
        fleet = self.extractor.fleet.get(self.variables.get("customer_number", {}).get("value") or "", set())
        if fleet and tail not in fleet:
            msg = f"Tail {tail} isn't on file for this customer. Confirm it with the caller."
            prompts.append(msg)
            self.prompts.append(msg)
            return []
        self.tail_slot.update(value=tail, partial="", partial_by="", pending=False)
        return ["tail_number"] if self._set("tail_number", tail, country, 0.8, text, speaker) else []

    def _settle(self, picks: dict, speaker: str) -> list[str]:
        """Apply Winnow's answers to this line's unsure matches (a pick of "none" keeps the fuzzy match)."""
        changed = []
        for var, pick in (picks or {}).items():
            if not isinstance(pick, dict):
                continue
            unsure = self._unsure.get(var)
            if unsure and pick["value"] in unsure["options"] and pick["p"] >= self.min_p:
                if self._set(var, pick["value"], unsure["options"][pick["value"]], pick["p"], unsure["heard"], speaker):
                    changed.append(var)
                self.variables[var]["settled_by"] = "winnow"
        return changed

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
        self._unsure = {}
        changed, prompts = self._extract(speaker, line["text"])
        self.prompts += prompts
        fired = self._autocall()
        decision = None
        if speaker in self.decide_on:
            # One model request: what the caller wants, plus which customer / location they meant if the match was fuzzy.
            ask_tail = (self.normalizer is not None and "tail_number" not in changed
                        and bool(slots.TAIL_TALK.search(line["text"]) or self.tail_slot["pending"] or self.tail_slot["partial"]))
            if ask_tail:
                self._unsure["tail_said"] = True
            picked = self.decider.decide(line["text"], self.transcript, self.variables, self._unsure)
            settled = self._settle(picked.get("variables"), speaker)
            if ask_tail and (picked.get("variables") or {}).get("tail_said", 0.0) >= self.min_p:
                settled += self._normalize_tail(line["text"], speaker, prompts)
            if settled:
                changed += [v for v in settled if v not in changed]
                fired += self._autocall()
            picked = {k: picked[k] for k in ("p", "by", "probabilities", "ms") if k in picked} | {"function": picked["function"]}
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
