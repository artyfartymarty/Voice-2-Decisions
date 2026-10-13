"""Glue: audio -> (diarization) -> Nemotron ASR -> roles -> winnow -> tool calls -> database.

Speakers are stored as the diarizer named them ("speaker 1", "speaker 2"; or the leg labels for two-leg
uploads). call.context["roles"] maps them to "agent" / "caller", and everything shown or sent to tools uses
the role.
"""
import logging
from dataclasses import dataclass

import numpy as np
from django.conf import settings
from django.db import transaction

from . import audio, extract, sentiment, tools
from .asr import get_backend
from .audio import TARGET_SR
from .diarize import close_live_session, get_diarizer, live_session
from .models import Call, Moment, Segment
from .roles import infer_roles
from .winnow import Winnower, _norm, summarize

log = logging.getLogger(__name__)
DEFAULT_SPEAKERS = ("agent", "caller")


@dataclass
class Piece:
    start: float
    end: float
    speaker: str
    text: str


def _winnower_for(call: Call) -> Winnower:
    return Winnower(
        pending=call.pending_text,
        seen={_norm(t) for t in call.moments.values_list("text", flat=True)},
        context=call.context,
        scorer=sentiment.get_scorer(),
        sentiment_window=settings.SENTIMENT_WINDOW,
        sentiment_alert_at=settings.SENTIMENT_ALERT_AT,
        dispute_min_count=settings.DISPUTE_MIN_COUNT,
    )


def _store_moments(call: Call, moments, roles: dict) -> list[Moment]:
    stored = []
    for m in moments:
        role = roles.get(m.speaker, m.speaker)
        if m.kind == "sentiment_shift":
            m.args["speaker"] = role
        if m.kind == "action_item" and m.speaker and m.owner == m.speaker:
            m.owner = role  # "I'll send it" said by speaker 1 -> owner: agent
        outcome = tools.dispatch(m.tool_call)
        stored.append(Moment.objects.create(
            call=call, kind=m.kind, text=m.text, score=m.score, start=m.start, end=m.end,
            owner=m.owner, due=m.due, also=m.also, speaker=m.speaker,
            tool_name=m.tool_call["name"], tool_args=m.tool_call["args"],
            tool_status=outcome["status"], tool_result=outcome["result"],
        ))
    return stored


def _transcribe_span(samples: np.ndarray, start: float, end: float, speaker: str, backend, langs: list | None = None) -> list[Piece]:
    a, b = int(start * TARGET_SR), int(end * TARGET_SR)
    t = backend.transcribe(samples[a:b])
    if langs is not None and t.language:
        langs.append(t.language)
    return [Piece(start + s.start, start + s.end, speaker, s.text) for s in t.segments if s.text]


def _roles_for(pieces: list[Piece], fixed: dict | None = None) -> dict:
    if fixed is not None:
        return {"roles": fixed, "scores": {}, "confident": True, "source": "legs"}
    info = infer_roles([(p.speaker, p.text) for p in pieces if p.speaker], extract.get())
    info["source"] = "diarization"
    return info


def _winnow_pieces(call: Call, pieces: list[Piece], roles: dict, by_turn: bool) -> list[Moment]:
    """Store transcript pieces and winnow them. by_turn: each piece is one speaker's complete turn."""
    w = _winnower_for(call)
    new = []
    for p in pieces:
        Segment.objects.create(call=call, start=p.start, end=p.end, text=p.text, speaker=p.speaker)
        new += _store_moments(call, w.feed(p.text, p.start, p.end, final=by_turn, speaker=p.speaker), roles)
    call.pending_text, call.context = w.pending, {**w.context, "roles": call.context.get("roles", {})}
    return new


def process_recording(call: Call, legs: list[bytes], speakers: list[str] | None = None) -> Call:
    """A finished recording.

    One file (the usual case: a mono call): diarize who spoke when, transcribe each turn, then name the
    speakers agent / caller from what they say. Two files (one per leg): each leg is its own speaker.
    LEG_MODE=mix folds two legs into one mono file first and treats it like a mono call.
    """
    try:
        decoded = [audio.decode(data) for data in legs]
        if len(decoded) > 1 and settings.LEG_MODE == "mix":
            decoded = [audio.mix_legs(decoded)]
        backend = get_backend()
        fixed_roles, langs = None, []
        if len(decoded) > 1:
            labels = [(speakers[i] if speakers and i < len(speakers) and speakers[i] else
                       (DEFAULT_SPEAKERS[i] if i < len(DEFAULT_SPEAKERS) else f"speaker {i + 1}")) for i in range(len(decoded))]
            pieces = [p for samples, label in zip(decoded, labels)
                      for p in _transcribe_span(samples, 0, len(samples) / TARGET_SR, label, backend, langs)]
            pieces.sort(key=lambda p: p.start)
            fixed_roles, by_turn = {label: label for label in labels}, True
        else:
            samples = decoded[0]
            diarizer = get_diarizer()
            turns = diarizer.diarize(samples) if diarizer else []
            if turns:
                pieces = []
                for t in turns:
                    span = _transcribe_span(samples, t.start, t.end, t.speaker, backend, langs)
                    if span:  # one piece per turn: the turn is the unit winnow judges
                        pieces.append(Piece(span[0].start, span[-1].end, t.speaker, " ".join(p.text for p in span)))
                by_turn = True
            else:
                pieces, by_turn = _transcribe_span(samples, 0, len(samples) / TARGET_SR, "", backend, langs), False
        role_info = _roles_for(pieces, fixed_roles)
        with transaction.atomic():
            call.context = {**call.context, "roles": role_info}
            _winnow_pieces(call, pieces, role_info["roles"], by_turn)
            w = _winnower_for(call)
            _store_moments(call, w.flush(pieces[-1].speaker if pieces else ""), role_info["roles"])
            call.pending_text, call.context = "", {**w.context, "roles": role_info}
            call.language = langs[0] if langs else ""
            call.duration = max(len(s) for s in decoded) / TARGET_SR
            call.status, call.pending_text = "done", ""
            call.save()
    except Exception as exc:
        log.exception("processing call %s failed", call.pk)
        call.status, call.error = "error", f"{type(exc).__name__}: {exc}"
        call.save(update_fields=["status", "error"])
    return call


def _split_chunk_by_turns(chunk_start: float, chunk_end: float, turns, last_speaker: str) -> list[tuple[float, float, str]]:
    """Cut [chunk_start, chunk_end] into (start, end, speaker) spans that follow the diarized turns.

    Audio the diarizer hasn't scored yet (its look-ahead at the end of the chunk) or silence between turns
    goes to the speaker before it.
    """
    spans = []
    for t in turns:
        a, b = max(t.start, chunk_start), min(t.end, chunk_end)
        if b - a > 0.05:
            spans.append([a, b, t.speaker])
    if not spans:
        return [(chunk_start, chunk_end, last_speaker)]
    spans[0][0] = chunk_start if spans[0][0] - chunk_start < 1.0 else spans[0][0]
    if spans[0][0] > chunk_start:
        spans.insert(0, [chunk_start, spans[0][0], last_speaker])
    for prev, nxt in zip(spans, spans[1:]):
        prev[1] = nxt[0]
    spans[-1][1] = chunk_end
    merged = []
    for a, b, s in spans:
        if merged and merged[-1][2] == s:
            merged[-1][1] = b
        else:
            merged.append([a, b, s])
    return [tuple(x) for x in merged]


def process_partial(call: Call, text: str = "", audio_bytes: bytes | None = None, final: bool = False,
                    speaker: str = "") -> list[Moment]:
    """One step of a live call: a new piece of transcript text, or a short audio chunk.

    Audio chunks are diarized with a streaming session that lives for the whole call, so "speaker 1" stays the
    same person from chunk to chunk; roles are re-inferred as evidence comes in. Decisions are made as soon as
    a sentence (or a speaker's turn) ends; an unfinished sentence waits for the next chunk.
    """
    backend = get_backend()
    pieces: list[Piece] = []
    chunk_start = call.duration
    if audio_bytes:
        samples = audio.decode(audio_bytes)
        chunk_end = chunk_start + len(samples) / TARGET_SR
        session = live_session(call.pk)
        if session is not None:
            turns = session.push(samples)
            if final:
                turns = session.finish()
                close_live_session(call.pk)
            last = call.context.get("last_speaker", "")
            for a, b, spk in _split_chunk_by_turns(chunk_start, chunk_end, turns, last):
                rel = samples[int((a - chunk_start) * TARGET_SR): int((b - chunk_start) * TARGET_SR)]
                t = backend.transcribe(rel)
                if t.text:
                    pieces.append(Piece(a, b, spk, t.text))
        else:
            t = backend.transcribe(samples)
            if t.text:
                pieces.append(Piece(chunk_start, chunk_end, speaker, t.text))
        call.duration = chunk_end
    if text:
        pieces.append(Piece(call.duration, call.duration, speaker, text))
    with transaction.atomic():
        if pieces:
            call.context = {**call.context, "last_speaker": pieces[-1].speaker}
        all_turns = [(s.speaker, s.text) for s in call.segments.all()] + [(p.speaker, p.text) for p in pieces]
        if any(s for s, _ in all_turns):
            info = infer_roles([t for t in all_turns if t[0]], extract.get())
            info["source"] = "diarization" if audio_bytes and get_diarizer() else "client"
            call.context["roles"] = info
        roles = call.context.get("roles", {}).get("roles", {})
        # A change of speaker ends the previous speaker's sentence.
        new = []
        prev = call.context.get("last_fed_speaker")
        for i, p in enumerate(pieces):
            if prev is not None and p.speaker != prev and call.pending_text:
                w = _winnower_for(call)
                new += _store_moments(call, w.flush(prev), roles)
                call.pending_text, call.context = "", {**w.context, "roles": call.context.get("roles", {})}
            new += _winnow_pieces(call, [p], roles, by_turn=False)
            prev = p.speaker
        call.context["last_fed_speaker"] = prev
        if final:
            w = _winnower_for(call)
            new += _store_moments(call, w.flush(prev or ""), roles)
            call.pending_text, call.context = "", {**w.context, "roles": call.context.get("roles", {})}
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
    summary.update(call_id=call.pk, title=str(call), status=call.status, duration=round(call.duration, 2),
                   language=call.language, speakers=call.roles, speaker_roles_confident=call.context.get("roles", {}).get("confident"))
    return summary


def call_payload(call: Call) -> dict:
    roles = call.roles
    return {
        "id": call.pk, "title": str(call), "status": call.status, "error": call.error,
        "language": call.language, "duration": round(call.duration, 2), "created_at": call.created_at.isoformat(),
        "transcript": call.transcript, "pending_text": call.pending_text, "context": call.context, "speakers": roles,
        "segments": [{"start": s.start, "end": s.end, "speaker": s.speaker, "role": roles.get(s.speaker, s.speaker), "text": s.text}
                     for s in call.segments.all()],
        "moments": [m.as_dict() for m in call.moments.all()],
    }
