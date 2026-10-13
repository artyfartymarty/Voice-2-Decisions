"""Golden calls: known calls with the tool calls they must (and must not) trigger. Spec: tests/golden/calls.json.

play_script() feeds a call's script as text partials (no speech model, for the fast tests). play_wav() streams its
WAV through the configured speech model and diarizer in live-sized chunks, the way the bridge and the test page do
(for `manage.py doctor --golden` on the GPU box). Both run inside a transaction that is rolled back, so nothing is
left in the database. compare() turns the result into a list of failures; empty means the call behaved.
"""
import io
import json
import wave
from pathlib import Path

from django.conf import settings
from django.db import transaction

from . import pipeline
from .models import Call

SPEC = Path(settings.BASE_DIR) / "tests" / "golden" / "calls.json"


class _Rollback(Exception):
    pass


def load(path: Path = SPEC) -> list[dict]:
    return json.loads(Path(path).read_text())["calls"]


def _result(call: Call) -> dict:
    roles = call.roles
    return {
        "tools": [{"tool": m.tool_name, "args": m.tool_args, "status": m.tool_status} for m in call.moments.all()],
        "roles": sorted(set(roles.values())),
        "warnings": call.context.get("warnings", []),
        "transcript": [f"{roles.get(s.speaker, s.speaker) or '?'}: {s.text}" for s in call.segments.all()],
    }


def _play(case: dict, feed) -> dict:
    out = {}
    try:
        with transaction.atomic():
            call = Call.objects.create(title=f"golden {case['name']}", status="live")
            pipeline.reset_live_state(call.pk)
            try:
                feed(call)
                call.refresh_from_db()
                out = _result(call)
            finally:
                pipeline.reset_live_state(call.pk)
            raise _Rollback
    except _Rollback:
        pass
    return out


def play_script(case: dict) -> dict:
    def feed(call):
        lines = case["script"]
        for i, (speaker, text) in enumerate(lines):
            pipeline.process_partial(call, text=text, speaker=speaker, final=i == len(lines) - 1)
    return _play(case, feed)


def wav_chunks(path: Path, seconds: float):
    """A WAV file as consecutive WAV files of `seconds` each (the last may be shorter)."""
    with wave.open(str(path), "rb") as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        while frames := w.readframes(int(rate * seconds)):
            buf = io.BytesIO()
            with wave.open(buf, "wb") as out:
                out.setnchannels(channels)
                out.setsampwidth(width)
                out.setframerate(rate)
                out.writeframes(frames)
            yield buf.getvalue()


def play_wav(case: dict, chunk_s: float = 2.0) -> dict:
    def feed(call):
        chunks = list(wav_chunks(Path(settings.BASE_DIR) / case["wav"], chunk_s))
        for i, data in enumerate(chunks):
            pipeline.process_partial(call, audio_bytes=data, final=i == len(chunks) - 1)
    return _play(case, feed)


def compare(case: dict, result: dict) -> list[str]:
    failures = []
    fired = result["tools"]
    for want in case.get("expect", []):
        hits = [t for t in fired if t["tool"] == want["tool"]
                and all(str(t["args"].get(k)) == str(v) for k, v in want.get("args", {}).items())]
        if len(hits) != want.get("times", len(hits) or 1):
            seen = [t["args"] for t in fired if t["tool"] == want["tool"]]
            failures.append(f"{want['tool']} {want.get('args', {})}: expected {want.get('times', 'at least 1')}x, "
                            f"got {len(hits)}x" + (f" (it fired with {seen})" if seen and not hits else ""))
    for name in case.get("forbid", []):
        if any(t["tool"] == name for t in fired):
            failures.append(f"{name} fired but must not")
    failures += [f"tool {t['tool']} ended {t['status']}" for t in fired if t["status"] != "done"]
    missing = set(case.get("roles", [])) - set(result["roles"])
    if missing:
        failures.append(f"no {', '.join(sorted(missing))} found among speakers (got {result['roles'] or 'none'})")
    failures += [f"transcript check: {w['message'].rstrip('.')}" for w in result["warnings"]]
    return failures
