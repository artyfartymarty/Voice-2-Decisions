"""The external model that pulls tail numbers and customer names out of what the caller said (GPT 6 Luna on Bedrock,
or any OpenAI-compatible endpoint). Off unless EXTRACT_MODEL_URL is set; then it is the main path for both (see
EXTRACT_MODE and the README notes).

The model is only called when Winnow says the caller's line contains the thing (a `noul` question in the same request
as the intent), and what comes back is still checked before it becomes a variable: a tail must be a valid registration
and, when the customer has a fleet on file, one of their tails; a customer name must match a customer in
customers.csv. The agent is prompted to read either back.

Two kinds of endpoint:
    EXTRACT_MODEL set     OpenAI-compatible chat: POST <url>/chat/completions; the reply's content is the value
    EXTRACT_MODEL empty   plain JSON: POST <url> {"kind": "tail_number"|"customer", "text", "recent"} -> {"value": "..."}
EXTRACT_MODEL_API_KEY is sent as a Bearer token (an Amazon Bedrock API key works). EXTRACT_MODEL_TIMEOUT defaults to
2 s. EXTRACT_MODEL_REASONING, when set, is sent as reasoning_effort ("none" is fastest on GPT 6 Luna).
"""
import json
import logging
import re
import urllib.error
import urllib.request

log = logging.getLogger(__name__)
PROMPTS = {
    "tail_number": "Convert the aircraft tail number (registration) the caller said into its written form, e.g. "
                   "\"November one two three alpha bravo\" -> N123AB, \"golf alpha bravo charlie delta\" -> GABCD. "
                   "If they corrected themselves, use the corrected one. Reply with the registration only, no dashes, "
                   "or NONE if there isn't a complete one.",
    "customer": "Which company or customer account did the caller say they are calling from or about? Reply with the "
                "company name as they said it (or the account number, as digits, if that's all they gave), nothing "
                "else, or NONE if they didn't say one.",
}


class ModelExtractor:
    def __init__(self, url: str, model: str = "", api_key: str = "", timeout: float = 2.0, reasoning: str = ""):
        self.url, self.model, self.api_key, self.timeout = url.rstrip("/"), model, api_key, timeout
        self.reasoning = reasoning

    def _post(self, url: str, body: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def extract(self, kind: str, text: str, recent: list[str]) -> str | None:
        try:
            if self.model:
                body = {"model": self.model,
                        # reasoning tokens count against this, so leave room unless reasoning is off
                        "max_completion_tokens": 24 if self.reasoning in ("", "none") else 1024,
                        "messages": [{"role": "system", "content": PROMPTS[kind]},
                                     {"role": "user", "content": "\n".join(recent[-3:] + [text])}]}
                if self.reasoning:
                    body["reasoning_effort"] = self.reasoning
                raw = self._post(f"{self.url}/chat/completions", body)["choices"][0]["message"]["content"]
            else:
                out = self._post(self.url, {"kind": kind, "text": text, "recent": recent[-3:]})
                raw = out.get("value") or out.get(kind) or ""
        except urllib.error.HTTPError as exc:  # e.g. Bedrock's reason for a 400 / 403
            log.warning("extract model failed (%s): %s %s", kind, exc, exc.read()[:300])
            return None
        except Exception as exc:
            log.warning("extract model failed (%s): %s", kind, exc)
            return None
        raw = str(raw or "").strip().strip("\"'.")
        if not raw or raw.upper() == "NONE":
            return None
        if kind == "tail_number":
            tail = re.sub(r"[^A-Z0-9]", "", raw.upper())
            return tail if 2 <= len(tail) <= 10 else None
        # A company name: letters, digits, spaces and & . , ' - only, at most 80 characters; anything else is refused.
        if len(raw) > 80 or not re.fullmatch(r"[A-Za-z0-9 &.,'\-]+", raw):
            log.warning("extract model answer refused (%s): %.80r", kind, raw)
            return None
        return raw


def from_env(env) -> ModelExtractor | None:
    url = env.get("EXTRACT_MODEL_URL", "")
    if not url:
        return None
    return ModelExtractor(url, env.get("EXTRACT_MODEL", ""), env.get("EXTRACT_MODEL_API_KEY", ""),
                          float(env.get("EXTRACT_MODEL_TIMEOUT", "2")), env.get("EXTRACT_MODEL_REASONING", ""))
