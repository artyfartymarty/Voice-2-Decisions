"""Name diarized speakers: which one is the agent, which the caller.

Diarization only says "speaker 1" and "speaker 2". On an inbound call the roles show in what people say:
the agent greets ("thanks for calling", "how can I help"), looks things up and gives prices; the caller
names their company or account, says where they're flying and asks for a quote. Each speaker's lines are
scored against both cue lists, plus a small prior that whoever speaks first answered the phone (the agent).
"""
import re

_I = re.IGNORECASE
AGENT_CUES = [
    (re.compile(r"\b(thanks?|thank you) for (calling|holding|waiting)\b", _I), 3.0),
    (re.compile(r"\bhow (can|may|could) i (help|assist)\b", _I), 3.0),
    (re.compile(r"\b(let me|i'll|i will|i can) (pull|look|check|get|bring) (that|this|it|you)?\s*(up|for you|into)?\b", _I), 2.0),
    (re.compile(r"\b(i can offer|we can offer|our (price|rate)|the (price|rate) (is|would be)|i('ll| will) send you the quote)\b", _I), 2.0),
    (re.compile(r"\b(is there anything else|anything else i can|have a (great|good) (day|flight))\b", _I), 2.0),
    (re.compile(r"\b(i understand|i apologi[sz]e|sorry (about|for) (that|the))\b", _I), 1.0),
    (re.compile(r"\b(can i (get|have) your|what('s| is) your (account|customer|tail))\b", _I), 2.0),
]
CALLER_CUES = [
    (re.compile(r"\b(i|we) (need|want|would like|'d like)( to get)? (a |some )?(quote|fuel|price|pricing)\b", _I), 3.0),
    (re.compile(r"\b(what('s| is) your (price|rate)|how much (is|are|for|per)|can i get a (quote|price))\b", _I), 3.0),
    (re.compile(r"\b(we('re| are) (flying|landing|arriving|coming)|our (aircraft|plane|jet|tail)|tail number)\b", _I), 2.0),
    (re.compile(r"\b(my|our) account\b|\baccount (number )?\d", _I), 2.0),
    (re.compile(r"\b(i'?m|i am) calling (about|to|for|from)\b", _I), 2.0),
    (re.compile(r"\b(last time (we|you)|you (guys )?(charged|overcharged|billed))\b", _I), 1.5),
]
FIRST_SPEAKER_PRIOR = 1.0


def cue_scores(text: str) -> tuple[float, float]:
    agent = sum(w * len(p.findall(text)) for p, w in AGENT_CUES)
    caller = sum(w * len(p.findall(text)) for p, w in CALLER_CUES)
    return agent, caller


def infer_roles(turns: list[tuple[str, str]], reference=None) -> dict:
    """turns: (speaker, text) in time order -> {"roles": {speaker: role}, "scores": {...}, "confident": bool}.

    With two or more speakers, the one with the strongest agent-minus-caller evidence is the agent, the most
    caller-like other speaker is the caller, and any others keep their diarized names.
    """
    speakers = list(dict.fromkeys(s for s, _ in turns))
    if not speakers:
        return {"roles": {}, "scores": {}, "confident": False}
    scores = {s: [0.0, 0.0] for s in speakers}
    for s, text in turns:
        a, c = cue_scores(text)
        if reference is not None and reference.find_customer(text):
            c += 2.0  # naming the customer company or account number is what callers do
        scores[s][0] += a
        scores[s][1] += c
    scores[speakers[0]][0] += FIRST_SPEAKER_PRIOR
    if len(speakers) == 1:
        a, c = scores[speakers[0]]
        role = "agent" if a > c else "caller" if c > a else speakers[0]
        return {"roles": {speakers[0]: role}, "scores": scores, "confident": abs(a - c) >= 2}
    margin = {s: scores[s][0] - scores[s][1] for s in speakers}
    agent = max(speakers, key=lambda s: margin[s])
    caller = min((s for s in speakers if s != agent), key=lambda s: margin[s])
    roles = {s: s for s in speakers}
    roles[agent], roles[caller] = "agent", "caller"
    gap = margin[agent] - margin[caller]
    return {"roles": roles, "scores": scores, "confident": gap >= 3}
