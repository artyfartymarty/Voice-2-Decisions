"""Optional fallback for tail numbers the parser couldn't read: an external model turns the caller's words into a
registration. Off unless TAIL_NORMALIZER_URL is set.

It is only asked when Winnow says the caller has just said a complete tail number (a `noul` question in the same
request as the intent) and the deterministic parser found none in that line. Whatever comes back is still checked
against the registration formats and, when the customer has a fleet on file, against that list, so a model can't put
an invented tail into a function call.

Two kinds of endpoint:
    TAIL_NORMALIZER_MODEL set     OpenAI-compatible chat: POST <url>/chat/completions, the reply's content is the tail
    TAIL_NORMALIZER_MODEL empty   plain JSON: POST <url> {"text", "recent"} -> {"tail_number": "N123AB"}
TAIL_NORMALIZER_API_KEY is sent as a Bearer token (an Amazon Bedrock API key works). TAIL_NORMALIZER_TIMEOUT defaults
to 2 s. TAIL_NORMALIZER_REASONING, when set, is sent as reasoning_effort. GPT 6 Luna on Bedrock: see the README.
"""
import json
import logging
import re
import urllib.error
import urllib.request

log = logging.getLogger(__name__)
PROMPT = ("Convert the aircraft tail number (registration) the caller said into its written form, e.g. "
          "\"November one two three alpha bravo\" -> N123AB, \"golf alpha bravo charlie delta\" -> GABCD. "
          "Reply with the registration only, no dashes, or NONE if there isn't a complete one.")


class TailNormalizer:
    def __init__(self, url: str, model: str = "", api_key: str = "", timeout: float = 2.0, reasoning: str = ""):
        self.url, self.model, self.api_key, self.timeout = url.rstrip("/"), model, api_key, timeout
        self.reasoning = reasoning  # sent as reasoning_effort when set ("none" for GPT 6 Luna: fastest)

    def _post(self, url: str, body: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def normalize(self, text: str, recent: list[str]) -> str | None:
        try:
            if self.model:
                body = {"model": self.model,
                        # reasoning tokens count against this, so leave room unless reasoning is off
                        "max_completion_tokens": 16 if self.reasoning in ("", "none") else 1024,
                        "messages": [{"role": "system", "content": PROMPT},
                                     {"role": "user", "content": "\n".join(recent[-3:] + [text])}]}
                if self.reasoning:
                    body["reasoning_effort"] = self.reasoning
                out = self._post(f"{self.url}/chat/completions", body)
                raw = out["choices"][0]["message"]["content"]
            else:
                raw = self._post(self.url, {"text": text, "recent": recent[-3:]}).get("tail_number") or ""
        except urllib.error.HTTPError as exc:  # e.g. Bedrock's reason for a 400 / 403
            log.warning("tail normalizer failed: %s %s", exc, exc.read()[:300])
            return None
        except Exception as exc:
            log.warning("tail normalizer failed: %s", exc)
            return None
        tail = re.sub(r"[^A-Z0-9]", "", str(raw).upper())
        return tail if tail and tail != "NONE" else None


def from_env(env) -> TailNormalizer | None:
    url = env.get("TAIL_NORMALIZER_URL", "")
    if not url:
        return None
    return TailNormalizer(url, env.get("TAIL_NORMALIZER_MODEL", ""), env.get("TAIL_NORMALIZER_API_KEY", ""),
                          float(env.get("TAIL_NORMALIZER_TIMEOUT", "2")), env.get("TAIL_NORMALIZER_REASONING", ""))
