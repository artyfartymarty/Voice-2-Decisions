"""Allowlist in front of winnow-server: only the typed-decision endpoint and health get through. Standard library only.

winnow-server also serves free-text generation (/v1/chat/completions), inspection and metrics. This service only needs
POST /v1/systemone, so the server listens on 127.0.0.1 inside the container and this guard is the only way in: every
other method or path gets 403, bodies over MAX_BODY get 413, and nothing but the decision JSON is passed back.
"""
import http.client
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ALLOWED = {("POST", "/v1/systemone"), ("GET", "/health"), ("GET", "/v1/health")}
MAX_BODY = 64 * 1024


def handler(upstream_host: str, upstream_port: int):
    class Guard(BaseHTTPRequestHandler):
        def _deny(self, code: int, why: str):
            data = json.dumps({"error": why}).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _forward(self):
            path = self.path.split("?")[0]
            if (self.command, path) not in ALLOWED:
                return self._deny(403, f"{self.command} {path} is not allowed; only POST /v1/systemone")
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                self.rfile.read(min(length, 16 * MAX_BODY))  # drain it, or Windows resets the connection mid-reply
                self.close_connection = True
                return self._deny(413, "request too large")
            body = self.rfile.read(length) if length else None
            headers = {k: v for k, v in self.headers.items() if k.lower() in ("content-type", "authorization")}
            conn = http.client.HTTPConnection(upstream_host, upstream_port, timeout=60)
            try:
                conn.request(self.command, path, body=body, headers=headers)
                resp = conn.getresponse()
                data = resp.read()
            except OSError as exc:
                return self._deny(502, f"winnow-server unavailable: {exc}")
            finally:
                conn.close()
            self.send_response(resp.status)
            self.send_header("Content-Type", resp.getheader("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_HEAD = _forward

        def log_message(self, *a):
            pass

    return Guard


def serve(host: str, port: int, upstream_host: str, upstream_port: int):
    ThreadingHTTPServer((host, port), handler(upstream_host, upstream_port)).serve_forever()
