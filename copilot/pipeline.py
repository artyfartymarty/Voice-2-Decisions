"""Glue: audio -> Nemotron ASR -> winnow -> tool calls -> database."""
import logging

from django.db import transaction

from . import audio, tools
from .asr import get_backend
from .models import Call, Moment, Segment
from .winnow import Winnower, _norm, summarize

log = logging.getLogger(__name__)


def _store_moments(call: Call, moments) -> list[Moment]:
    stored = []
    for m in moments:
        outcome = tools.dispatch(m.tool_call)
        stored.append(Moment.objects.create(
            call=call, kind=m.kind, text=m.text, score=m.score, start=m.start, end=m.end,
            owner=m.owner, due=m.due, also=m.also, tool_name=m.tool_call["name"], tool_args=m.tool_call["args"],
            tool_status=outcome["status"], tool_result=outcome["result"],
        ))
    return stored


def _winnower_for(call: Call) -> Winnower:
    return Winnower(pending=call.pending_text, seen={_norm(t) for t in call.moments.values_list("text", flat=True)})


def process_recording(call: Call, legs: list[bytes]) -> Call:
    """A finished recording, as one file or one file per leg (agent, caller): mix to mono, transcribe, winnow."""
    try:
        samples = audio.mix_legs([audio.decode(data) for data in legs])
        transcript = get_backend().transcribe(samples)
        with transaction.atomic():
            w = _winnower_for(call)
            for seg in transcript.segments:
                Segment.objects.create(call=call, start=seg.start, end=seg.end, text=seg.text)
                _store_moments(call, w.feed(seg.text, seg.start, seg.end))
            _store_moments(call, w.flush())
            call.language, call.duration, call.status, call.pending_text = transcript.language, transcript.duration, "done", ""
            call.save()
    except Exception as exc:
        log.exception("processing call %s failed", call.pk)
        call.status, call.error = "error", f"{type(exc).__name__}: {exc}"
        call.save(update_fields=["status", "error"])
    return call


def process_partial(call: Call, text: str = "", audio_bytes: bytes | None = None, final: bool = False) -> list[Moment]:
    """One step of a live call: a new piece of transcript text, or a short audio chunk to transcribe first.

    Decisions are made on partial transcripts as soon as a sentence ends; the unfinished tail waits for more.
    """
    start = call.duration
    if audio_bytes:
        samples = audio.decode(audio_bytes)
        chunk = get_backend().transcribe(samples)
        text = f"{text} {chunk.text}".strip()
        call.duration += chunk.duration
        call.language = call.language or chunk.language
    with transaction.atomic():
        if text:
            Segment.objects.create(call=call, start=start, end=max(call.duration, start), text=text)
        w = _winnower_for(call)
        new = _store_moments(call, w.feed(text, start, call.duration, final=final))
        call.pending_text = w.pending
        call.status = "done" if final else "live"
        call.save()
    return new


def replay_pending(call: Call | None = None) -> int:
    """Re-run tool calls that were waiting for a tool to be registered. Returns how many now succeeded."""
    qs = Moment.objects.filter(tool_status="pending")
    if call is not None:
        qs = qs.filter(call=call)
    done = 0
    for m in qs:
        outcome = tools.dispatch({"name": m.tool_name, "args": m.tool_args})
        if outcome["status"] != "pending":
            m.tool_status, m.tool_result = outcome["status"], outcome["result"]
            m.save(update_fields=["tool_status", "tool_result"])
            done += outcome["status"] == "done"
    return done


def call_summary(call: Call) -> dict:
    summary = summarize([m.as_dict() for m in call.moments.all()])
    summary.update(call_id=call.pk, title=str(call), status=call.status, duration=round(call.duration, 2), language=call.language)
    return summary


def call_payload(call: Call) -> dict:
    return {
        "id": call.pk, "title": str(call), "status": call.status, "error": call.error,
        "language": call.language, "duration": round(call.duration, 2), "created_at": call.created_at.isoformat(),
        "transcript": call.transcript, "pending_text": call.pending_text,
        "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in call.segments.all()],
        "moments": [m.as_dict() for m in call.moments.all()],
    }
