"""winnow: sift a call transcript down to the things that matter.

Each finished sentence is scored against rule sets for five kinds of moment:

    decision     "we decided to go with the annual plan"
    action_item  "Sarah will send the contract by Friday"
    risk         "my concern is the onboarding timeline"
    question     "can you confirm the start date?"
    key_moment   prices, budgets, deadlines, cancellations

Every moment carries the tool call it maps to (see tools.py), so downstream code can act on it.
Winnower keeps state between partial transcripts: it only judges sentences once they end,
holds the unfinished tail until more text arrives, and never reports the same sentence twice.
"""
import re
from dataclasses import asdict, dataclass, field

KINDS = ("decision", "action_item", "risk", "question", "key_moment")

TOOL_FOR_KIND = {
    "decision": "log_decision",
    "action_item": "create_action_item",
    "risk": "flag_risk",
    "question": "open_question",
    "key_moment": "mark_key_moment",
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
        (re.compile(r"\b(i'll|i will|we'll|we will|you'll|you will|[A-Z][a-z]+ will)\s+(send|email|call|schedule|set up|share|draft|prepare|review|update|check|book|follow|get|create|write|loop)\b"), 0.85),
        (re.compile(r"\b(can|could|would) you (please )?(send|email|call|schedule|set up|share|draft|prepare|review|update|check|book|follow|get|create|write)\b", _I), 0.7),
        (re.compile(r"\b(need(s)? to|have to|must)\s+(send|email|call|schedule|set up|share|draft|prepare|review|update|check|book|follow|get|create|write)\b", _I), 0.6),
    ],
    "risk": [
        (re.compile(r"\b(concern(ed)?|worried|worry|risk|blocker|blocked|problem|issue|objection|hesitant)\b", _I), 0.75),
        (re.compile(r"\b(too (expensive|slow|risky|late)|won't work|doesn't work|can't afford|not sure (we|this)|push ?back)\b", _I), 0.8),
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

    @property
    def tool_call(self) -> dict:
        args = {"text": self.text}
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


class Winnower:
    """Incremental winnowing over a stream of partial transcripts."""

    def __init__(self, pending: str = "", seen: set[str] | None = None):
        self.pending = pending
        self.seen = set(seen or ())

    def feed(self, text: str, start: float | None = None, end: float | None = None, final: bool = False) -> list[Moment]:
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
                    moments.append(moment)
        return moments

    def flush(self) -> list[Moment]:
        return self.feed("", final=True)


def winnow(text: str) -> list[Moment]:
    """Whole-transcript convenience wrapper."""
    w = Winnower()
    return w.feed(text, final=True)


def summarize(moments: list[dict]) -> dict:
    """Summary of a call from its moments (dicts with at least kind/text/owner/due)."""
    by_kind = {k: [m for m in moments if m["kind"] == k] for k in KINDS}
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
    }
