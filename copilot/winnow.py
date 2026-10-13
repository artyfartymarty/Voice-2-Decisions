"""winnow: sift a call transcript down to the things that matter.

Each finished sentence is scored against rule sets for five kinds of moment:

    decision     "we decided to go with the annual plan"
    action_item  "Sarah will send the contract by Friday"
    risk         "my concern is the onboarding timeline"
    question     "can you confirm the start date?"
    key_moment   prices, budgets, deadlines, cancellations

On top of those, Winnower tracks the call as a whole (customer, location, tail number, whether a quote was
asked for, rolling sentiment) and raises the use-case moments:

    quote_request    a quote is wanted and both customer and location are known  -> quote_lookup
    dispute_risk     a quote is wanted at a location with a dispute history       -> dispute_guidance
    sentiment_shift  the conversation turns negative                              -> alert_manager
    tail_number      a complete, confident registration (see slots.py)            -> validate_tail_number
    agent_prompt     something the agent should confirm (e.g. a cut-off tail)     -> prompt_agent

winnow decides *which* tool fires; the tool arguments (customer number, ICAO, tail number) come from the
deterministic extractor in extract.py, so they are always values from the reference data or a strict grammar.

Every moment carries the tool call it maps to (see tools.py and usecases.py), so downstream code can act on it.
Winnower keeps state between partial transcripts: it only judges sentences once they end,
holds the unfinished tail until more text arrives, and never reports the same sentence twice.
"""
import re
from dataclasses import asdict, dataclass, field

from . import slots

KINDS = ("decision", "action_item", "risk", "question", "key_moment")
USE_CASE_KINDS = ("quote_request", "dispute_risk", "sentiment_shift", "tail_number", "agent_prompt")
ALL_KINDS = KINDS + USE_CASE_KINDS

TOOL_FOR_KIND = {
    "decision": "log_decision",
    "action_item": "create_action_item",
    "risk": "flag_risk",
    "question": "open_question",
    "key_moment": "mark_key_moment",
    "quote_request": "quote_lookup",
    "dispute_risk": "dispute_guidance",
    "sentiment_shift": "alert_manager",
    "tail_number": "validate_tail_number",
    "agent_prompt": "prompt_agent",
}

_I = re.IGNORECASE
RULES: dict[str, list[tuple[re.Pattern, float]]] = {
    "decision": [
        (re.compile(r"\b(we|i)(\s+have|'ve)?\s+decided\b", _I), 0.9),
        (re.compile(r"\b(decision is|final decision|we('re| are) going (with|to go with)|let's go with|we('ll| will) go with)\b", _I), 0.9),
        (re.compile(r"\b((we|they|both|everyone|all)( have| all)? agreed|agreed (to|on)|we agree|(it's|that's) a deal|signed off|sign off on|approved?|green ?light(ed)?)\b", _I), 0.7),
        (re.compile(r"\b(let's|we('ll| will)) (do|use|move forward|proceed|choose|pick|switch)\b", _I), 0.6),
    ],
    "action_item": [
        (re.compile(r"\b(action item|follow[- ]up|next step)\b", _I), 0.8),
        (re.compile(r"\b((?i:i'll|i will|we'll|we will|you'll|you will)|[A-Z][a-z]+ will)\s+(?i:send|email|call|schedule|set up|share|draft|prepare|review|update|check|book|follow|get|create|write|loop)\b"), 0.85),
        (re.compile(r"\b(can|could|would) you (please )?(send|email|call|schedule|set up|share|draft|prepare|review|update|check|book|follow|get|create|write)\b", _I), 0.7),
        (re.compile(r"\b(need(s)? to|have to|must)\s+(send|email|call|schedule|set up|share|draft|prepare|review|update|check|book|follow|get|create|write)\b", _I), 0.6),
    ],
    "risk": [
        (re.compile(r"\b(concern(ed)?|worried|worry|risk|blocker|blocked|problem|issue|objection|hesitant)\b", _I), 0.75),
        (re.compile(r"\b(too (expensive|slow|risky|late)|won't work|doesn't work|can't afford|not sure (we|this)|push ?back)\b", _I), 0.8),
        # complaints are a risk to the account
        (re.compile(r"\b(overcharg\w*|unacceptable|frustrat\w*|disappoint\w*|complain\w*|dispute\w*|ripped off)\b", _I), 0.75),
    ],
    "key_moment": [
        (re.compile(r"[$€£]\s?\d[\d,]*(\.\d+)?\s?(k|m|million|thousand)?\b", _I), 0.85),
        (re.compile(r"\b\d[\d,]*(\.\d+)?\s?(%|percent|dollars|euros|k\b|thousand|million)", _I), 0.75),
        # ASR often spells amounts out: "forty thousand dollars", "fifteen percent"
        (re.compile(r"\b(hundred|thousand|million|billion|grand)\s+(dollars|euros|pounds|bucks)\b|\b[a-z]+[- ]?(percent)\b", _I), 0.75),
        (re.compile(r"\b(budget|pricing|price|discount|deadline|go[- ]live|launch date|cancel(lation)?|churn|competitor)\b", _I), 0.6),
    ],
}

OWNER = re.compile(r"\b(I|we|you|[A-Z][a-z]+)(?:'ll| will| can| should| needs? to| has to)\b")
DUE = re.compile(
    r"\b(by|before|on|until|due)\s+((?:next|this)\s+\w+|tomorrow|today|tonight|end of (?:the )?(?:day|week|month|quarter)|eod|eow"
    r"|monday|tuesday|wednesday|thursday|friday|saturday|sunday"
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?"
    r"|\d{1,2}/\d{1,2}(?:/\d{2,4})?)",
    _I,
)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=[.!?])$")
# Streaming ASR often drops sentence punctuation ("...the annual plan Sarah will send..."),
# so a sentence is also cut where a new clause with its own subject starts.
CLAUSE_BREAK = re.compile(
    r"(?<=,)\s+(?=(?:(?i:and|but|so)\s+)?(?:(?i:we|i|they|you|he|she)\b|[A-Z][a-z]+\s))"
    r"|\s+(?=[A-Z][a-z]+\s+(?:will|is going to|needs to|has to|should)\b)"
)
MIN_CLAUSE_WORDS = 3
MIN_SCORE = 0.55

QUOTE_INTENT = re.compile(r"\b(quotes?|quoted|quoting|pric(e|es|ing)|how much|per gallon|rates?|cost)\b", _I)
SENTIMENT = [
    (re.compile(r"\b(frustrat\w*|annoy\w*|angry|furious|upset|unacceptable|ridiculous|terrible|awful|horrible|disappoint\w*"
                r"|overcharg\w*|ripped off|rip-?off|waste of|worst|complain\w*|fed up|sick of|never again|not happy|unhappy|hate)\b", _I), -1.0),
    (re.compile(r"\b(problem|wrong|late|delay(ed)?|too expensive|confus\w*|mistake)\b", _I), -0.5),
    (re.compile(r"\b(great|perfect|excellent|appreciate\w*|thank(s| you)|wonderful|happy|love|awesome|fantastic|sounds good|glad|helpful)\b", _I), 1.0),
]


def sentiment(text: str) -> float:
    """Lexicon score in [-1, 1]; 0 means no sentiment words."""
    total = sum(weight * len(pat.findall(text)) for pat, weight in SENTIMENT)
    return max(-1.0, min(1.0, total))


@dataclass
class Moment:
    kind: str
    text: str
    score: float
    start: float | None = None
    end: float | None = None
    owner: str = ""
    due: str = ""
    also: list[str] = field(default_factory=list)
    args: dict = field(default_factory=dict)  # extra tool arguments for use-case moments
    speaker: str = ""  # "agent", "caller"... when the call came in as separate legs

    @property
    def tool_call(self) -> dict:
        args = {"text": self.text, **self.args}
        if self.kind == "action_item":
            args.update(owner=self.owner, due=self.due)
        if self.start is not None:
            args["at"] = round(self.start, 2)
        return {"name": TOOL_FOR_KIND[self.kind], "args": args}

    def as_dict(self) -> dict:
        d = asdict(self)
        d["tool_call"] = self.tool_call
        return d


def split_sentences(text: str) -> tuple[list[str], str]:
    """Split into finished sentences and the unfinished tail (text after the last . ! or ?)."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return [], ""
    parts = [p.strip() for p in SENTENCE_END.split(text) if p and p.strip()]
    if parts and not re.search(r"[.!?][\"')\]]*$", parts[-1]):
        return parts[:-1], parts[-1]
    return parts, ""


def split_clauses(sentence: str) -> list[str]:
    """Cut a sentence where a new subject starts; fragments under MIN_CLAUSE_WORDS stay with their neighbour."""
    pieces = [p.strip() for p in CLAUSE_BREAK.split(sentence) if p and p.strip()]
    clauses: list[str] = []
    for piece in pieces:
        if clauses and (len(piece.split()) < MIN_CLAUSE_WORDS or len(clauses[-1].split()) < MIN_CLAUSE_WORDS):
            clauses[-1] = f"{clauses[-1]} {piece}"
        else:
            clauses.append(piece)
    return clauses or [sentence]


def score_sentence(sentence: str) -> dict[str, float]:
    scores = {}
    for kind, rules in RULES.items():
        best = max((w for pat, w in rules if pat.search(sentence)), default=0.0)
        if best:
            scores[kind] = best
    if sentence.rstrip().endswith("?"):
        scores["question"] = max(scores.get("question", 0.0), 0.8)
    return scores


def judge(sentence: str, start: float | None = None, end: float | None = None) -> Moment | None:
    """One sentence -> its strongest moment (others go in .also), or None if nothing scores high enough."""
    scores = {k: v for k, v in score_sentence(sentence).items() if v >= MIN_SCORE}
    if not scores:
        return None
    # On ties, prefer the more actionable kind (order of KINDS).
    kind = max(scores, key=lambda k: (scores[k], -KINDS.index(k)))
    moment = Moment(kind=kind, text=sentence, score=scores[kind], start=start, end=end,
                    also=[k for k in KINDS if k in scores and k != kind])
    if kind == "action_item":
        owner = OWNER.search(sentence)
        moment.owner = owner.group(1) if owner and owner.group(1).lower() not in {"i", "we", "you"} else (owner.group(1) if owner else "")
        due = DUE.search(sentence)
        moment.due = due.group(2) if due else ""
    return moment


def _norm(sentence: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", sentence.lower()).strip()


def new_context() -> dict:
    return {"customer": None, "customer_name": None, "icao": None, "tail_number": None, "quote_intent": False,
            "quoted": [], "dispute_warned": [], "tail": slots.new_slot(), "recent": {}, "alerted": {}}


class Winnower:
    """Incremental winnowing over a stream of partial transcripts.

    `context` is JSON-serializable call state (who, where, quote asked for, recent sentiment); the pipeline
    stores it on the Call so a live call keeps it between requests.
    """

    def __init__(self, pending: str = "", seen: set[str] | None = None, context: dict | None = None,
                 reference=None, sentiment_window: int = 3, sentiment_alert_at: float = -0.4, dispute_min_count: int = 1,
                 extractor=None):
        self.pending = pending
        self.seen = set(seen or ())
        self.context = {**new_context(), **(context or {})}
        self._reference = reference
        self._extractor = extractor
        self.sentiment_window = sentiment_window
        self.sentiment_alert_at = sentiment_alert_at
        self.dispute_min_count = dispute_min_count

    @property
    def reference(self):
        if self._reference is None:
            from . import reference
            self._reference = reference.get()
        return self._reference

    @property
    def extractor(self):
        if self._extractor is None:
            from . import extract
            self._extractor = extract.get()
        return self._extractor

    def _use_cases(self, clause: str, start, end, speaker: str = "") -> list[Moment]:
        ctx, out = self.context, []
        found = self.extractor.extract(clause)
        if "customer" in found:
            ctx["customer"], ctx["customer_name"] = found["customer"].value, found["customer"].label
        if "location" in found:
            ctx["icao"] = found["location"].value
        for event in slots.update(ctx["tail"], clause, speaker, self.extractor, ctx["customer"]):
            if event["type"] == "validate":
                ctx["tail_number"] = event["tail_number"]
                out.append(Moment("tail_number", clause, event["confidence"], start, end, speaker=speaker,
                                  args={k: event[k] for k in ("tail_number", "country", "confidence", "heard", "how", "replaces")}
                                  | {"customer_number": ctx["customer"]}))
            else:
                out.append(Moment("agent_prompt", clause, 0.7, start, end, speaker=speaker,
                                  args={"message": event["message"], "about": "tail_number",
                                        "suggestion": event.get("suggestion")}))
        if QUOTE_INTENT.search(clause):
            ctx["quote_intent"] = True

        if ctx["quote_intent"] and ctx["icao"]:
            pair = f"{ctx['customer']}:{ctx['icao']}"
            if ctx["customer"] and pair not in ctx["quoted"]:
                ctx["quoted"].append(pair)
                out.append(Moment("quote_request", clause, 0.9, start, end, speaker=speaker,
                                  args={"customer_number": ctx["customer"], "icao": ctx["icao"]}))
            loc = self.reference.locations.get(ctx["icao"])
            if loc and loc.dispute_count >= self.dispute_min_count and ctx["icao"] not in ctx["dispute_warned"]:
                ctx["dispute_warned"].append(ctx["icao"])
                out.append(Moment("dispute_risk", clause, 0.85, start, end, speaker=speaker, args={"icao": ctx["icao"]}))

        score = sentiment(clause)
        if score:
            # Tracked per speaker, so a friendly agent doesn't mask an angry caller (and vice versa).
            recent = (ctx["recent"].get(speaker, []) + [[score, clause]])[-self.sentiment_window:]
            ctx["recent"][speaker] = recent
            scores = [s for s, _ in recent]
            avg = sum(scores) / len(scores)
            turned = len(scores) >= 2 and scores[-1] < 0 and scores[-2] < 0
            if not ctx["alerted"].get(speaker) and (turned or (sum(s < 0 for s in scores) >= 2 and avg <= self.sentiment_alert_at)):
                ctx["alerted"][speaker] = True
                out.append(Moment("sentiment_shift", clause, round(-avg, 2), start, end, speaker=speaker,
                                  args={"speaker": speaker, "avg_sentiment": round(avg, 2), "recent": [t for _, t in recent]}))
            elif ctx["alerted"].get(speaker) and avg >= 0:
                ctx["alerted"][speaker] = False  # recovered: a later downturn alerts again
        return out

    def feed(self, text: str, start: float | None = None, end: float | None = None, final: bool = False,
             speaker: str = "") -> list[Moment]:
        sentences, tail = split_sentences(f"{self.pending} {text}")
        if final and tail:
            sentences, tail = sentences + [tail], ""
        self.pending = tail
        moments = []
        for sentence in sentences:
            for clause in split_clauses(sentence):
                key = _norm(clause)
                if not key or key in self.seen:
                    continue
                self.seen.add(key)
                moment = judge(clause, start, end)
                if moment:
                    moment.speaker = speaker
                    if moment.kind == "action_item" and speaker and moment.owner in ("", "I"):
                        moment.owner = speaker  # "I'll send it", said by the agent -> owner: agent
                    moments.append(moment)
                moments.extend(self._use_cases(clause, start, end, speaker))
        return moments

    def flush(self, speaker: str = "") -> list[Moment]:
        return self.feed("", final=True, speaker=speaker)


def winnow(text: str) -> list[Moment]:
    """Whole-transcript convenience wrapper."""
    w = Winnower()
    return w.feed(text, final=True)


def summarize(moments: list[dict]) -> dict:
    """Summary of a call from its moments (dicts with at least kind/text/owner/due)."""
    by_kind = {k: [m for m in moments if m["kind"] == k] for k in ALL_KINDS}
    headline = by_kind["decision"][0]["text"] if by_kind["decision"] else (
        by_kind["action_item"][0]["text"] if by_kind["action_item"] else "No decisions or action items detected.")
    return {
        "headline": headline,
        "counts": {k: len(v) for k, v in by_kind.items()},
        "decisions": [m["text"] for m in by_kind["decision"]],
        "action_items": [{"text": m["text"], "owner": m.get("owner", ""), "due": m.get("due", "")} for m in by_kind["action_item"]],
        "risks": [m["text"] for m in by_kind["risk"]],
        "open_questions": [m["text"] for m in by_kind["question"]],
        "key_moments": [m["text"] for m in by_kind["key_moment"]],
        "quotes": [m.get("tool_result") or m["tool_call"]["args"] for m in by_kind["quote_request"]],
        "dispute_warnings": [m.get("tool_result") or m["tool_call"]["args"] for m in by_kind["dispute_risk"]],
        "sentiment_alerts": [m.get("tool_result") or m["tool_call"]["args"] for m in by_kind["sentiment_shift"]],
        "tail_numbers": [m.get("tool_result") or m["tool_call"]["args"] for m in by_kind["tail_number"]],
        "agent_prompts": [m["tool_call"]["args"]["message"] for m in by_kind["agent_prompt"]],
    }
