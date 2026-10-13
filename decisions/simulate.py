"""Pretend to be the Nova Sonic side: stream a scripted call into the service, chunk by chunk, in real time.

Each line of the script becomes the events Nova Sonic sends for one turn: a SPECULATIVE text block that grows a few
words at a time while the person is speaking, then the FINAL text block (contentStart -> textOutput -> contentEnd,
USER for the caller and ASSISTANT for the agent). The variables, function calls and decisions print as they happen.

    python -m decisions.simulate demos/demo_quote_call.json                       # in-process, real time
    python -m decisions.simulate --url http://localhost:8000 demos/*.json         # against the running service
    python -m decisions.simulate --speed 4 demos/demo_frustrated_caller.json      # 4x faster (0 = no waiting)
    python -m decisions.simulate --dump demos/demo_quote_call.json > call.jsonl   # just print the events, send nothing

A script is a JSON list of turns: {"speaker": "caller" | "agent", "text": "..."}, with optional "start" and "end" in
seconds from the start of the call. Without them a turn takes as long as its words at about 2.5 words a second, and the
next starts 0.6 s later. Add a call by dropping a new file like that into demos/.
"""
import argparse
import itertools
import json
import os
import sys
import time
import urllib.request

WORDS_PER_SECOND = 2.5
GAP = 0.6
ROLE = {"caller": "USER", "agent": "ASSISTANT"}


def timed(turns: list[dict]) -> list[dict]:
    """Fill in start / end where the script leaves them out."""
    out, clock = [], 0.5
    for t in turns:
        start = t.get("start", clock)
        end = t.get("end", start + max(1.0, len(t["text"].split()) / WORDS_PER_SECOND))
        out.append({**t, "start": start, "end": end})
        clock = end + GAP
    return out


def nova_events(turn: dict, ids, chunk: int = 3, partials: bool = True):
    """(seconds from call start, event) pairs for one turn, shaped like Nova Sonic's output stream."""
    role, words = ROLE.get(turn["speaker"], turn["speaker"].upper()), turn["text"].split()

    def block(stage, text, at, stop="PARTIAL_TURN"):
        cid = f"c{next(ids)}"
        fields = json.dumps({"generationStage": stage})
        return [(at, {"event": {"contentStart": {"contentId": cid, "type": "TEXT", "role": role,
                                                 "additionalModelFields": fields}}}),
                (at, {"event": {"textOutput": {"contentId": cid, "role": role, "content": text}}}),
                (at, {"event": {"contentEnd": {"contentId": cid, "type": "TEXT", "stopReason": stop}}})]

    events = []
    if partials and len(words) > chunk:
        steps = range(chunk, len(words), chunk)
        span = turn["end"] - turn["start"]
        for n in steps:
            events += block("SPECULATIVE", " ".join(words[:n]), turn["start"] + span * n / len(words))
    return events + block("FINAL", turn["text"], turn["end"], "END_TURN")


def show(step: dict, live: bool):
    if not step["final"]:
        if live:
            print(f"\r\033[K   ...  {step['preview'][:110]}", end="", flush=True)
        return
    if live:
        print("\r\033[K", end="")
    print(f"{step['line']['speaker']:>7}: {step['line']['text']}")
    for name in step["variables_changed"]:
        print(f"         variable  {name}")
    for f in step["fired"]:
        print(f"         called    {f['function']}({', '.join(map(str, f['args'].values()))})")
    for prompt in step["prompts"]:
        print(f"         prompt    {prompt}")
    if step["decision"]:
        d = step["decision"]
        extra = "".join(f", {k}={d[k]}" for k in ("p", "ms") if k in d)
        print(f"      >> DECISION  {d['function']} [{d['by']}{extra}]: {d['say']}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scripts", nargs="+", help="call scripts (.json), played one after another")
    p.add_argument("--url", help="running service, e.g. http://localhost:8000 (default: in-process)")
    p.add_argument("--speed", type=float, default=1.0, help="1 = real time, 4 = four times faster, 0 = no waiting")
    p.add_argument("--chunk", type=int, default=3, help="words added per speculative update (default 3)")
    p.add_argument("--no-partials", action="store_true", help="send only the FINAL text of each turn")
    p.add_argument("--dump", action="store_true", help="print the events as JSON lines instead of sending them")
    a = p.parse_args()

    send = None
    if a.url:
        def send(call_id, ev):
            req = urllib.request.Request(f"{a.url.rstrip('/')}/calls/{call_id}/events", data=json.dumps(ev).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
    elif not a.dump:
        from .server import App
        app = App(os.environ)
        send = app.events
    live = sys.stdout.isatty()

    for path in a.scripts:
        call_id = f"sim-{os.path.splitext(os.path.basename(path))[0]}-{int(time.time())}"
        ids = itertools.count(1)
        timeline = [e for t in timed(json.load(open(path))) for e in nova_events(t, ids, a.chunk, not a.no_partials)]
        if not a.dump:
            print(f"=== {path} (call {call_id})")
        t0 = time.monotonic()
        for at, ev in timeline:
            if a.speed > 0:
                time.sleep(max(0.0, at / a.speed - (time.monotonic() - t0)))
            if a.dump:
                print(json.dumps(ev))
                continue
            for step in send(call_id, ev)["steps"]:
                show(step, live)
        if not a.dump:
            print()


if __name__ == "__main__":
    main()
