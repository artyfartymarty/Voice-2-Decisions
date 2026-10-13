"""Turn incoming transcript events into (speaker, text, final) lines.

Accepted shapes, one object or a list of them:

    Amazon Nova Sonic, raw output events from the bidirectional stream:
        {"event": {"contentStart": {"contentId": "c1", "type": "TEXT", "role": "USER",
                                    "additionalModelFields": "{\"generationStage\":\"FINAL\"}"}}}
        {"event": {"textOutput": {"contentId": "c1", "role": "USER", "content": "I'd like a fuel quote"}}}
    Nova Sonic textOutput, flattened:
        {"role": "USER", "content": "I'd like a fuel quote", "generationStage": "FINAL"}
    Plain:
        {"speaker": "caller", "text": "I'd like a fuel quote"}

Nova's roles map to call speakers with ROLE_MAP (default USER -> caller, ASSISTANT -> agent). Only FINAL text is
parsed; SPECULATIVE text is returned as a preview and never fills variables or fires functions. Events without a
stage (plain ones, or textOutput whose contentStart we never saw) count as final.
"""
import json
import os

DEFAULT_ROLE_MAP = {"USER": "caller", "ASSISTANT": "agent"}


def role_map() -> dict[str, str]:
    raw = os.environ.get("ROLE_MAP", "")
    pairs = (p.split(":", 1) for p in raw.split(",") if ":" in p)
    return {**DEFAULT_ROLE_MAP, **{k.strip().upper(): v.strip().lower() for k, v in pairs}}


def _stage(fields) -> str:
    if isinstance(fields, str):
        try:
            fields = json.loads(fields)
        except ValueError:
            return ""
    return (fields or {}).get("generationStage", "") if isinstance(fields, dict) else ""


class EventReader:
    """Keeps the per-call contentId -> (role, stage) map that raw Nova streams need."""

    def __init__(self):
        self.content: dict[str, tuple[str, str]] = {}

    def read(self, payload) -> list[dict]:
        events = payload if isinstance(payload, list) else [payload]
        lines = []
        for ev in events:
            line = self._one(ev if isinstance(ev, dict) else {})
            if line and line["text"].strip():
                lines.append(line)
        return lines

    def _one(self, ev: dict) -> dict | None:
        inner = ev.get("event", ev)
        if "contentStart" in inner:
            cs = inner["contentStart"]
            if cs.get("contentId"):
                self.content[cs["contentId"]] = (cs.get("role", ""), _stage(cs.get("additionalModelFields")))
            return None
        if "contentEnd" in inner:
            self.content.pop(inner["contentEnd"].get("contentId", ""), None)
            return None
        if "textOutput" in inner:
            inner = inner["textOutput"]
        text = inner.get("content", inner.get("text"))
        if not isinstance(text, str):
            return None
        role, stage = self.content.get(inner.get("contentId", ""), ("", ""))
        role = inner.get("role", role)
        stage = (inner.get("generationStage") or _stage(inner.get("additionalModelFields")) or stage).upper()
        speaker = inner.get("speaker") or role_map().get(str(role).upper(), str(role).lower() or "caller")
        # Nova Sonic marks barge-in with this literal text; it isn't speech.
        if text.strip() == '{ "interrupted" : true }':
            return None
        return {"speaker": speaker, "text": text, "final": stage != "SPECULATIVE"}
