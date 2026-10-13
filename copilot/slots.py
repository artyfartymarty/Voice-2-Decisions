"""Per-call slot for the tail number, so it survives how people actually say it.

Callers spell registrations over several breaths, ASR cuts them off, agents re-ask, and callers correct:

    caller: "tail is bravo tango"                 -> partial "BT"; prompt the agent to get the full number
    agent:  "sorry, what was the tail number again?"   -> slot pending: the next spelled run is the answer
    caller: "November five one two bravo tango"   -> N512BT, complete and confident -> validate_tail_number
    caller: "no, that's B T, not B D"             -> correction applied to the current value
    caller: "November five one two" ... "bravo tango"  -> spread-out spelling is joined

The slot lives in the call context (JSON), so it works across live partials. A tool call fires only for a complete
registration at or above the confidence threshold (or one found in the customer's fleet). Anything short of that
becomes a prompt to the agent, never a guess passed to a tool.
"""
import re

from .extract import TAIL_CUE, registration_country, symbol_runs

REASK = re.compile(
    r"\b(sorry|pardon|come again|say (that|it) again|repeat (that|it)|one more time|didn'?t (catch|get) (that|it)"
    r"|what was (the|that|your)( tail)?( number)?( again)?|again\?)", re.IGNORECASE)
TAIL_TALK = re.compile(r"\b(tail|registration|aircraft|plane|jet)\b", re.IGNORECASE)
TAIL_WORDS = re.compile(r"\b(tail|registration|reg|number|that)\b", re.IGNORECASE)
# The caller fixing what they just said, mid-sentence: "bravo delta, sorry I meant bravo tango", "scratch that".
SELF_CORRECT = re.compile(r"\b(i mean|i meant|sorry|no wait|wait|scratch that|actually|correction|make that|rather"
                          r"|forget (?:that|it|the \w+))\b", re.IGNORECASE)
CORRECTION = re.compile(r"\b(no|not|correction|i mean|actually|sorry it'?s|that'?s)\b", re.IGNORECASE)


def new_slot() -> dict:
    return {"value": None, "partial": "", "partial_by": "", "pending": False, "fired": [], "prompted": []}


def _confidence(reg: str, clause: str, run, slot: dict) -> float:
    conf = 0.6
    if TAIL_CUE.search(clause) or slot["pending"]:
        conf += 0.25
    if run.phonetic or len(reg) >= 5:
        conf += 0.1
    if len(run.heard.split()) > 1:
        conf += 0.05
    return round(min(conf, 0.99), 2)


def update(slot: dict, clause: str, speaker: str, extractor, customer: str | None, _base: str | None = None) -> list[dict]:
    """Feed one clause. Returns events: {"type": "validate", ...} or {"type": "prompt", "message": ...}."""
    # A self-correction inside the clause ("bravo delta, sorry I meant bravo tango", "five one two, scratch that,
    # November eight eight..."): only what follows the last cue counts, so the abandoned version is never committed;
    # what came before it is the base a partial fix ("I meant bravo tango") is applied to.
    cues = list(SELF_CORRECT.finditer(clause)) if _base is None else []
    if cues and symbol_runs(clause[cues[-1].end():]):
        before = [r.symbols for r in symbol_runs(clause[:cues[-1].start()]) if r.phonetic or registration_country(r.symbols)]
        if before:
            return update(slot, clause[cues[-1].end():], speaker, extractor, customer, _base=max(before, key=len))
    if REASK.search(clause) and TAIL_WORDS.search(clause):
        slot["pending"] = True
    known = extractor.icaos | set(extractor.iata) | set(extractor.customers)
    runs = [r for r in symbol_runs(clause) if r.symbols not in known and (len(r.symbols) >= 2 or _base is not None)]
    # Only spelling-like runs count unless we're expecting a tail: avoids dates ("November three") and stray letters.
    # "November" alone is not tail talk (it's also a month); a correction right after a tail is.
    tail_context = (_base is not None or bool(TAIL_TALK.search(clause)) or slot["pending"] or bool(slot["partial"])
                    or bool(slot["value"] and CORRECTION.search(clause)))
    runs = [r for r in runs if r.phonetic or registration_country(r.symbols)]
    runs = [r for r in runs if tail_context or (registration_country(r.symbols) and len(r.symbols) >= 4)]
    if not runs:
        return []

    good = [r for r in runs if not r.negated]
    bad = [r for r in runs if r.negated]
    base = _base or slot["value"] or slot["partial"]
    run = max(good or runs, key=lambda r: len(r.symbols))
    candidate, how = run.symbols, "heard"

    if _base is not None and not registration_country(candidate) and len(candidate) < len(base) \
            and registration_country(base[: len(base) - len(candidate)] + candidate):
        candidate, how = base[: len(base) - len(candidate)] + candidate, "corrected"  # "...I meant bravo tango"
    elif _base is not None:
        how = "restated" if registration_country(candidate) else how
    elif bad and good and base and CORRECTION.search(clause):
        wrong, right = bad[0].symbols, good[0].symbols
        if wrong in base:
            candidate, how = base.replace(wrong, right, 1), "corrected"
        elif base.endswith(right[:-1]) and len(right) <= len(base):
            candidate, how = base[: len(base) - len(right)] + right, "corrected"
    elif slot["partial"] and not slot["pending"] and not registration_country(candidate) and speaker == slot["partial_by"]:
        candidate, how = slot["partial"] + candidate, "joined"  # spelling spread over several breaths
    elif slot["partial"] and not slot["pending"] and registration_country(slot["partial"] + candidate) and speaker == slot["partial_by"] \
            and not registration_country(slot["partial"]):
        candidate, how = slot["partial"] + candidate, "joined"

    events = []
    fleet = extractor.fleet.get(customer or "", set())
    # US callers often drop the leading "November": "five one two bravo tango" is N512BT. Only for a full-length
    # body or a tail on file, so a first breath ("five one two" ... "bravo tango") still waits for the rest.
    if not registration_country(candidate) and registration_country("N" + candidate) == "US" \
            and ("N" + candidate in fleet or len(candidate) >= 4):
        candidate, how = "N" + candidate, "n_implied"
    country = registration_country(candidate)
    if country:
        conf = 0.99 if candidate in fleet else _confidence(candidate, clause, run, slot)
        if how == "corrected":
            conf = max(conf, 0.9)  # fixing letters of a tail already confirmed in this call
        if fleet and candidate not in fleet:
            conf = min(conf, 0.75)
            msg = f"Tail {candidate} isn't on file for this customer. Confirm it with the caller."
            if msg not in slot["prompted"]:
                slot["prompted"].append(msg)
                events.append({"type": "prompt", "message": msg, "tail_number": candidate})
        if conf >= extractor.tail_min_confidence:
            if candidate not in slot["fired"]:
                slot["fired"].append(candidate)
                events.append({"type": "validate", "tail_number": candidate, "country": country, "confidence": conf,
                               "heard": run.heard, "how": how, "replaces": slot["value"] if slot["value"] != candidate else None})
            slot.update(value=candidate, partial="", partial_by="", pending=False)
            return events
        slot.update(partial=candidate, partial_by=speaker, pending=False)
        return events
    # Incomplete: keep it, and tell the agent what to ask for.
    slot.update(partial=candidate, partial_by=speaker, pending=False)
    suggestion = sorted(t for t in fleet if t.endswith(candidate) or candidate in t)
    msg = (f"Caller corrected the tail mid-sentence and it doesn't add up (heard \"{run.heard}\" after \"{_base}\"). "
           "Read the tail back to them and confirm it." if _base is not None else
           f"Tail number incomplete: heard \"{run.heard}\" ({candidate}). Ask the caller to repeat the full tail number.")
    if len(suggestion) == 1:
        msg += f" On file for this customer: {suggestion[0]}."
    if candidate not in slot["prompted"]:
        slot["prompted"].append(candidate)
        events.append({"type": "prompt", "message": msg, "partial": candidate, "suggestion": suggestion[0] if len(suggestion) == 1 else None})
    return events
