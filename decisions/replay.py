"""Feed a call to the service and print what it decided. For testing without Nova Sonic.

    python -m decisions.replay demos/demo_quote_call.json                      # a demo call
    python -m decisions.replay calls/nova_events.jsonl                          # recorded Nova Sonic events, one per line
    python -m decisions.replay -                                                # type lines: "caller: I need a quote"
    python -m decisions.replay --url http://localhost:8000 FILE                 # against a running server

Without --url it runs in-process (no server needed), with the decider from the environment (WINNOW_URL or rules).
"""
import argparse
import json
import os
import sys
import urllib.request


def lines_from(path: str):
    if path.endswith(".json"):
        for line in json.load(open(path)):  # [{"speaker", "text"}, ...]
            yield {"role": "USER" if line["speaker"] == "caller" else "ASSISTANT", "content": line["text"],
                   "generationStage": "FINAL"}
        return
    for raw in (sys.stdin if path == "-" else open(path)):
        raw = raw.strip()
        if not raw:
            continue
        if raw.startswith("{"):
            yield json.loads(raw)
        else:
            speaker, _, text = raw.partition(":") if ":" in raw else ("caller", "", raw)
            yield {"speaker": speaker.strip().lower(), "text": text.strip()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", help="a demo .json, .jsonl of events, a text file of 'speaker: text' lines, or - for stdin")
    p.add_argument("--url", help="running server, e.g. http://localhost:8000 (default: in-process)")
    p.add_argument("--call", default="replay")
    a = p.parse_args()
    if a.url:
        def send(ev):
            req = urllib.request.Request(f"{a.url.rstrip('/')}/calls/{a.call}/events", data=json.dumps(ev).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
    else:
        from .server import App
        app = App(os.environ)
        send = lambda ev: app.events(a.call, ev)  # noqa: E731
    for ev in lines_from(a.file):
        for step in send(ev)["steps"]:
            if not step["final"]:
                continue
            print(f"{step['line']['speaker']:>7}: {step['line']['text']}")
            for name in step["variables_changed"]:
                print(f"         variable  {name}")
            for f in step["fired"]:
                print(f"         called    {f['function']}({', '.join(map(str, f['args'].values()))})")
            for prompt in step["prompts"]:
                print(f"         prompt    {prompt}")
            if step["decision"]:
                d = step["decision"]
                print(f"      >> DECISION  {d['function']} [{d['by']}{', p=' + str(d['p']) if 'p' in d else ''}{', ' + str(d['ms']) + ' ms' if 'ms' in d else ''}]: {d['say']}")


if __name__ == "__main__":
    main()
