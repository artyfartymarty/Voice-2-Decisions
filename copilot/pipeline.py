"""Glue: audio -> (diarization) -> Nemotron ASR -> roles -> winnow -> tool calls -> database.

Speakers are stored as the diarizer named them ("speaker 1", "speaker 2"; or the leg labels for two-leg
uploads). call.context["roles"] maps them to "agent" / "caller", and everything shown or sent to tools uses
the role.
"""
import logging
import threading
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
            if getattr(backend, "supports_speakers", False) and diarizer is not None:
                # Speech service: ASR + diarization in one request; consecutive segments of a speaker form a turn.
                pieces = []
                for seg, spk in backend.transcribe_with_speakers(samples):
                    if pieces and pieces[-1].speaker == spk:
                        pieces[-1].end, pieces[-1].text = seg.end, f"{pieces[-1].text} {seg.text}"
                    else:
                        pieces.append(Piece(seg.start, seg.end, spk, seg.text))
                langs.append("en-US")
                by_turn = True
            elif diarizer is not None and (turns := diarizer.diarize(samples)):
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


_carry: dict[int, tuple[float, np.ndarray]] = {}  # live call id -> (call time of carried audio, samples)
_live_locks: dict[int, threading.Lock] = {}
_flush_timers: dict[int, threading.Timer] = {}
MAX_CARRY_S = 6.0


def _lock(call_id: int) -> threading.Lock:
    return _live_locks.setdefault(call_id, threading.Lock())


def _schedule_flush(call_id: int):
    """If audio is still held back and no new chunk arrives within LIVE_FLUSH_S, transcribe it anyway.

    Without this, the last words before a quiet stretch (or the end of a stream that never sends final) would
    only show up when the next chunk or the final request arrives.
    """
    old = _flush_timers.pop(call_id, None)
    if old:
        old.cancel()
    if call_id in _carry and settings.LIVE_FLUSH_S > 0:
        t = threading.Timer(settings.LIVE_FLUSH_S, flush_live_audio, args=(call_id, True))
        t.daemon = True
        _flush_timers[call_id] = t
        t.start()


def flush_live_audio(call_id: int, from_timer: bool = False) -> list:
    """Transcribe and winnow whatever audio a live call is holding back. Safe to call any time."""
    from django.db import close_old_connections

    if from_timer:
        close_old_connections()  # the timer runs on its own thread with its own DB connection
    try:
        call = Call.objects.filter(pk=call_id, status="live").first()
        if call is None or call_id not in _carry:
            return []
        return process_partial(call, flush=True)
    except Exception:
        log.exception("flushing live audio for call %s failed", call_id)
        return []
    finally:
        if from_timer:
            close_old_connections()


def reset_live_state(call_id: int):
    """Forget in-memory live state (carried audio, diarization session) for a call that is (re)starting."""
    timer = _flush_timers.pop(call_id, None)
    if timer:
        timer.cancel()
    _carry.pop(call_id, None)
    close_live_session(call_id)


def _cut_at_pause(buf: np.ndarray, final: bool) -> int:
    """How many samples of `buf` can be transcribed now: up to the middle of the last pause.

    Audio after that may be a word still being spoken. Continuous speech longer than MAX_CARRY_S is cut at its
    quietest point so latency stays bounded.
    """
    if final or not len(buf):
        return len(buf)
    spans = audio.utterances(buf)
    if not spans:
        return len(buf)  # only silence
    guard = int(0.25 * TARGET_SR)
    for i in range(len(spans) - 1, -1, -1):
        end = spans[i][1]
        if end < len(buf) - guard:  # this utterance is followed by a pause
            nxt = spans[i + 1][0] if i + 1 < len(spans) else len(buf)
            return (end + nxt) // 2
    if len(buf) > MAX_CARRY_S * TARGET_SR:
        return audio.split_on_silence(buf, max_seconds=MAX_CARRY_S)[0][1]
    return 0


def process_partial(call: Call, text: str = "", audio_bytes: bytes | None = None, final: bool = False,
                    speaker: str = "", flush: bool = False) -> list[Moment]:
    """Thread-safe entry: one live step at a time per call (requests and the flush timer)."""
    old = _flush_timers.pop(call.pk, None)
    if old:
        old.cancel()
    with _lock(call.pk):
        new = _process_partial(call, text, audio_bytes, final, speaker, flush)
    if not final:
        _schedule_flush(call.pk)
    return new


def _process_partial(call: Call, text: str, audio_bytes: bytes | None, final: bool, speaker: str,
                     flush: bool) -> list[Moment]:
    """One step of a live call: a new piece of transcript text, or a short audio chunk.

    Audio chunks are diarized with a streaming session that lives for the whole call, so "speaker 1" stays the
    same person from chunk to chunk; roles are re-inferred as evidence comes in. Decisions are made as soon as
    a sentence (or a speaker's turn) ends; an unfinished sentence waits for the next chunk.
    """
    backend = get_backend()
    pieces: list[Piece] = []
    chunk_start = call.duration
    if audio_bytes or (flush and call.pk in _carry):
        samples = audio.decode(audio_bytes) if audio_bytes else np.zeros(0, dtype=np.float32)
        chunk_end = chunk_start + len(samples) / TARGET_SR
        session = live_session(call.pk, create=bool(audio_bytes))
        turns = []
        if session is not None:
            turns = session.push(samples) if len(samples) else session.turns()
            if final:
                turns = session.finish()
                close_live_session(call.pk)
        # Transcribe only up to the last pause; the rest waits for the next chunk so no word is cut in half.
        carry_start, carry = _carry.pop(call.pk, (chunk_start, np.zeros(0, dtype=np.float32)))
        buf = np.concatenate([carry, samples])
        cut = _cut_at_pause(buf, final or flush)
        if cut < len(buf):
            _carry[call.pk] = (carry_start + cut / TARGET_SR, buf[cut:])
        ready_end = carry_start + cut / TARGET_SR
        if cut:
            spans = (_split_chunk_by_turns(carry_start, ready_end, turns, call.context.get("last_speaker", ""))
                     if session is not None else [(carry_start, ready_end, speaker)])
            for a, b, spk in spans:
                t = backend.transcribe(buf[int((a - carry_start) * TARGET_SR): int((b - carry_start) * TARGET_SR)])
                if t.text:
                    pieces.append(Piece(a, b, spk, t.text))
        call.duration = chunk_end
    if text:
        pieces.append(Piece(call.duration, call.duration, speaker, text))
    with transaction.atomic():
        if pieces:
            call.context = {**call.context, "last_speaker": pieces[-1].speaker}
        all_turns = [(s.speaker, s.text) for s in call.segments.all()] + [(p.speaker, p.text) for p in pieces]
        if any(s for s, _ in all_turns):
            info = infer_roles([t for t in all_turns if t[0]], extract.get())
            info["source"] = "diarization" if (audio_bytes or flush) and get_diarizer() else "client"
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
