"""HTTP API for the text-to-decisions service. Standard library only.

    POST   /calls/<id>/events   Nova Sonic text events (or {"speaker","text"}), one object or a list -> what happened
    GET    /calls/<id>          the call's variables, function results, current decision and transcript
    DELETE /calls/<id>          forget a call
    GET    /functions           the functions and the variables each one needs
    GET    /demos               the demo calls (tests/fixtures/*.script.json) the console can replay
    GET    /healthz             liveness + which decider is in use
    GET    /                    a small console: type caller / agent lines and watch it work

    python -m decisions.server            # listens on 0.0.0.0:8000 (PORT to change)

Settings (environment): REFERENCE_DIR, WINNOW_URL, WINNOW_API_KEY, DECIDER=auto|rules, DECIDE_MIN_P, ROLE_MAP,
DECISION_WEBHOOK_URL (each new decision is POSTed there as JSON).
"""
import json
import logging
import os
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import decider as decider_mod
from .engine import Call, load_extractor
from .events import EventReader
from .normalizer import from_env as normalizer_from_env
from .functions import FUNCTIONS

log = logging.getLogger("decisions")
ROOT = Path(__file__).resolve().parent
CONSOLE = (ROOT / "console.html").read_bytes()
DEMOS = sorted((ROOT.parent / "tests" / "fixtures").glob("*.script.json"))


class App:
    def __init__(self, env=os.environ):
        ref_dir = env.get("REFERENCE_DIR") or str(ROOT.parent / "fixtures" / "reference")
        self.extractor = load_extractor(ref_dir, float(env.get("TAIL_MIN_CONFIDENCE", "0.8")))
        self.decider = decider_mod.from_env(env)
        self.min_p = float(env.get("DECIDE_MIN_P", "0.5"))
        self.normalizer = normalizer_from_env(env)
        self.webhook = env.get("DECISION_WEBHOOK_URL", "")
        self.calls: dict[str, tuple[Call, EventReader, threading.Lock]] = {}
        self.lock = threading.Lock()

    def call(self, call_id: str):
        with self.lock:
            if call_id not in self.calls:
                self.calls[call_id] = (Call(call_id, self.extractor, self.decider, self.min_p, normalizer=self.normalizer), EventReader(), threading.Lock())
            return self.calls[call_id]

    def events(self, call_id: str, payload) -> dict:
        call, reader, lock = self.call(call_id)
        with lock:  # lines of one call are handled in order
            out = [call.feed(line["speaker"], line["text"], line["final"]) for line in reader.read(payload)]
        for step in out:
            if step.get("decision") and self.webhook:
                threading.Thread(target=self._post, args=({"call_id": call_id, **step["decision"]},), daemon=True).start()
        return {"call_id": call_id, "steps": out, "decision": call.decision, "waiting_for": call.waiting_for,
                "variables": {k: v["value"] for k, v in call.variables.items()}}

    def _post(self, body: dict):
        req = urllib.request.Request(self.webhook, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            urllib.request.urlopen(req, timeout=3).close()
        except Exception as exc:
            log.warning("decision webhook failed: %s", exc)


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _call_id(self):
            parts = self.path.split("?")[0].strip("/").split("/")
            return parts[1] if len(parts) >= 2 and parts[0] == "calls" else None

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                return self._send(200, CONSOLE, "text/html; charset=utf-8")
            if path == "/healthz":
                return self._send(200, {"ok": True, "decider": app.decider.name, "calls": len(app.calls)})
            if path == "/demos":
                return self._send(200, {p.name.split(".")[0]: json.loads(p.read_text()) for p in DEMOS})
            if path == "/functions":
                return self._send(200, [{"name": f.name, "needs": f.needs, "optional": f.optional, "auto": f.auto,
                                         "description": f.description} for f in FUNCTIONS])
            call_id = self._call_id()
            if call_id and call_id in app.calls:
                return self._send(200, app.calls[call_id][0].state())
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            call_id = self._call_id()
            if not call_id or not self.path.split("?")[0].endswith("/events"):
                return self._send(404, {"error": "POST /calls/<id>/events"})
            try:
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"null")
            except ValueError:
                return self._send(400, {"error": "body must be JSON"})
            return self._send(200, app.events(call_id, payload))

        def do_DELETE(self):
            call_id = self._call_id()
            with app.lock:
                found = app.calls.pop(call_id, None) if call_id else None
            return self._send(200 if found else 404, {"deleted": bool(found)})

        def log_message(self, fmt, *args):
            log.info("%s %s", self.address_string(), fmt % args)

    return Handler


def main():
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
    app = App()
    port = int(os.environ.get("PORT", "8000"))
    log.info("decisions listening on :%d (decider: %s)", port, app.decider.name)
    ThreadingHTTPServer(("0.0.0.0", port), make_handler(app)).serve_forever()


if __name__ == "__main__":
    main()
