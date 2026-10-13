"""One call: transcript lines in; variables, auto-called functions and the decision out.

    line ──► variables ──► functions whose `needs` are all filled are called ──► results wait in the call
      │                                                                              │
      └─(caller, final)─► decision model: what is the caller asking for? ───────────┴─► surface that result

If the caller asks for something whose variables aren't filled yet, the decision says what to ask for, and the result
is surfaced on its own as soon as the function fires (no second model call).

Variables come from the deterministic extractor (extract.py, slots.py), so
only values that exist in the reference data, or a valid registration, can fill one:

    customer_number   account number or company name, fuzzy + phonetic ("North Wind Aviation" -> 204511)
    icao              ICAO / IATA, spelled or phonetic, FBO or airport name ("Van Nuys" -> KVNY)
    tail_number       registration over several breaths, re-asks and corrections ("November eight eight ..." -> N88NW)
"""
import re
import time
from difflib import SequenceMatcher

from . import extract, slots

from .functions import BY_NAME, FUNCTIONS

SENTENCE = re.compile(r"(?<=[.!?])\s+")


class Call:
    def __init__(self, call_id: str, extractor, decider, min_p: float = 0.5, decide_on=("caller",), model=None,
                 mode: str = "model"):
        self.id = call_id
        self.extractor, self.decider, self.model = extractor, decider, model
        # "model": tail numbers and customer names come from the external model (Winnow says one was said -> the model
        # pulls it out -> checked against the formats / customers.csv / fleet -> read back). Account numbers said as
        # digits are still matched directly. "parser": the list matching and tail parser (extract.py, slots.py). Without a model
        # configured, "model" can't run, so the parser is used.
        self.by_model = mode == "model" and model is not None
        self._name_hint = False  # this line has something the list matcher took for a customer name
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
        self._ambiguous: set[str] = set()  # the customer came from a name with look-alikes: these are the options
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
            if "customer" in found and self.by_model and not re.fullmatch(r"\d+", found["customer"].evidence):
                self._name_hint = True  # a name: the model reads it (see _model_customer)
            elif "customer" in found:
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
            if self.by_model:
                continue
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

    def _recent(self) -> list[str]:
        return [f"{l['speaker']}: {l['text']}" for l in self.transcript[-4:-1]]

    def _prompt(self, msg: str, prompts: list[str]):
        prompts.append(msg)
        self.prompts.append(msg)

    def _choose(self, what: str, heard: str, text: str, options: dict) -> dict | None:
        """Winnow picks which of the close matches the caller meant. None: Winnow unavailable (or the rules decider)."""
        return self.decider.choose(what, heard, text, self.transcript, options) if options else None

    def _fits(self, numbers) -> list[str]:
        """The customers among `numbers` that agree with the tail and airport heard so far in the call."""
        tail = self.variables.get("tail_number", {}).get("value")
        icao = self.variables.get("icao", {}).get("value")
        return [n for n in numbers
                if (tail is None or tail in self.extractor.fleet.get(n, ()))
                and (icao is None or icao in {site.icao for site in self.extractor.ref.sites_for(n)})]

    def _model_customer(self, text: str, speaker: str, prompts: list[str]) -> list[str]:
        """Winnow says the caller named their company: the model pulls the name out, then it must match customers.csv.

        A transcript can be wrong in two ways. The name isn't on file ("Arigo"): the customers that look or sound
        closest are the options. Or it lands on another real customer ("Arrow" when they said "Aero"): an exact match
        on a name with look-alikes (Extractor.confusable) is treated as ambiguous too. Ambiguous options are narrowed by
        the tail and airport already heard; if more than one is left, Winnow chooses with the call so far. Either way
        the agent reads the account back, and a later tail or airport can still switch it (_recheck_customer)."""
        said = self.model.extract("customer", text, self._recent())
        if not said:
            return []
        exact = self.extractor.exact_customer(said)
        if exact is not None:
            options = [exact] + sorted(self.extractor.confusable[exact])
        else:
            options = [n for n, _, _ in self.extractor.closest_customers(said)]
        number, p, how = exact, 1.0, ""
        if len(options) > 1 or exact is None:
            fit = self._fits(options)
            if len(fit) == 1 and len(options) > 1:
                number, p, how = fit[0], 1.0, "context"
            else:
                pick = self._choose("customer", said, text, {n: self.extractor.describe(n) for n in (fit or options)})
                if pick and pick["value"] and pick["p"] >= self.min_p:
                    number, p, how = pick["value"], pick["p"], "winnow"
                elif pick is None and exact is None and (e := self.extractor.customer(said)):  # no Winnow: fuzzy match
                    number, p = e.value, e.confidence
                elif pick is not None or exact is None:
                    self._prompt(f"Caller said they're with \"{said}\", which doesn't match a customer on file. "
                                 f"Ask for the account number.", prompts)
                    return []
        self._ambiguous = set(options) if len(options) > 1 else set()
        name = self.extractor.customers[number][0]
        if not self._set("customer_number", number, name, p, said, speaker):
            return []
        if how:
            self.variables["customer_number"]["settled_by"] = how
        alike = [self.extractor.customers[n][0] for n in sorted(self._ambiguous - {number})]
        warn = (f" It sounds like {', '.join(alike)}; if the caller doesn't confirm, ask for the account number."
                if alike else "")
        self._prompt(f"Confirm the account: caller said \"{said}\", matched {name} ({number}).{warn}", prompts)
        return ["customer_number"]

    def _recheck_customer(self, speaker: str, prompts: list[str]) -> list[str]:
        """The customer came from a name with look-alikes, and a tail or airport now fits exactly one other of them:
        switch to that one and have the agent confirm."""
        current = self.variables.get("customer_number", {}).get("value")
        if not self._ambiguous or current is None:
            return []
        fit = self._fits(self._ambiguous)
        if current in fit or len(fit) != 1:
            return []
        new, old = fit[0], self.extractor.customers[current][0]
        heard = self.variables["customer_number"]["heard"]
        self._set("customer_number", new, self.extractor.customers[new][0], 1.0, heard, speaker)
        self.variables["customer_number"]["settled_by"] = "context"
        why = " and ".join(f"{k} {self.variables[v]['value']}" for k, v in (("tail", "tail_number"), ("airport", "icao"))
                           if v in self.variables)
        self._prompt(f"Account changed: the caller's {why} fits {self.extractor.customers[new][0]} ({new}), not {old}. "
                     f"Confirm the account with the caller.", prompts)
        return ["customer_number"]

    def _normalize_tail(self, text: str, speaker: str, prompts: list[str]) -> list[str]:
        """Winnow says the caller said a full tail: the model writes it out, then it must be valid and on the fleet.
        A tail that isn't on the customer's fleet goes to Winnow with the fleet's closest tails ("N47JA": N47AJ ...)."""
        tail = self.model.extract("tail_number", text, self._recent())
        country = extract.registration_country(tail or "")
        if not country:
            return []
        fleet = self.extractor.fleet.get(self.variables.get("customer_number", {}).get("value") or "", set())
        heard, how, p = tail, "", 0.8
        lookalike_fleet = any(tail in self.extractor.fleet.get(n, ()) for n in self._ambiguous)  # see _recheck_customer
        if fleet and tail not in fleet and not lookalike_fleet:
            off = {t: extract.edits(t, tail) for t in fleet}  # one or two letters or digits misheard or swapped
            close = [t for t in sorted(off, key=off.get)[:4] if off[t] <= 2]
            pick = self._choose("aircraft tail number", tail, text, {t: t for t in close})
            if not (pick and pick["value"] and pick["p"] >= self.min_p):
                self._prompt(f"Tail {tail} isn't on file for this customer. Confirm it with the caller.", prompts)
                return []
            tail, how, p = pick["value"], "winnow", pick["p"]
            country = extract.registration_country(tail) or country
        self.tail_slot.update(value=tail, partial="", partial_by="", pending=False)
        if not self._set("tail_number", tail, country, p, text, speaker):
            return []
        if how:
            self.variables["tail_number"]["settled_by"] = how
        said = f" (heard {heard})" if heard != tail else ""
        self._prompt(f"Read the tail number back to the caller: {tail}{said}.", prompts)
        return ["tail_number"]

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
        self._unsure, self._name_hint = {}, False
        changed, prompts = self._extract(speaker, line["text"])
        self.prompts += prompts
        changed += [v for v in self._recheck_customer(speaker, prompts) if v not in changed]
        fired = self._autocall()
        decision = None
        if speaker in self.decide_on:
            # One model request: what the caller wants, plus which customer / location they meant if the match was fuzzy.
            agent_asked = any(l["speaker"] != speaker and slots.TAIL_TALK.search(l["text"]) for l in self.transcript[-2:-1])
            ask_tail = (self.by_model and "tail_number" not in changed
                        and bool(slots.TAIL_TALK.search(line["text"]) or agent_asked
                                 or self.tail_slot["pending"] or self.tail_slot["partial"]))
            # Customer by name: asked until the customer is known, or when the line sounds like a (new) name.
            ask_customer = self.by_model and "customer_number" not in changed and (
                "customer_number" not in self.variables or self._name_hint)
            if ask_tail:
                self._unsure["tail_said"] = True
            if ask_customer:
                self._unsure["customer_said"] = {"hint": self._name_hint}
            picked = self.decider.decide(line["text"], self.transcript, self.variables, self._unsure)
            said = picked.get("variables") or {}
            settled = self._settle(said, speaker)
            if ask_customer and said.get("customer_said", 0.0) >= self.min_p:
                settled += self._model_customer(line["text"], speaker, prompts)
            if ask_tail and said.get("tail_said", 0.0) >= self.min_p:
                settled += self._normalize_tail(line["text"], speaker, prompts)
            settled += self._recheck_customer(speaker, prompts)
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
