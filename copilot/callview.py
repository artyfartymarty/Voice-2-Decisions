"""What the call page shows a call handler, built from a Call's moments and context.

Kept apart from the views so the live panels (refreshed while the call runs) and the first page load render the
same data, and so it can be tested without a request.
"""
import json
import zlib
from datetime import timedelta

import soundfile as sf
from django.conf import settings

from . import reference
from .models import Call, Moment
from .usecases import ALERTS

# Prompt cards on the call page, in the order a call handler reads them. Each use-case kind maps to its sheet
# Category; tail checks and agent prompts go under "Confirm on the call".
CARD_GROUPS = (
    ("GP", ("quote_request",)),
    ("Avoiding Disputes", ("dispute_risk",)),
    ("Customer Experience", ("sentiment_shift",)),
    ("Confirm on the call", ("agent_prompt", "tail_number")),
)
USE_CASE_KINDS = tuple(k for _, kinds in CARD_GROUPS for k in kinds)
# Moments that reach someone beyond the call handler: shown on the manager alert view.
ALERT_KINDS = ("sentiment_shift", "quote_request")


def who_is_calling(call: Call) -> dict:
    """The who's-calling card: customer, location, tail and quote state, from what winnow has filled in so far."""
    ctx = call.context or {}
    ref = reference.get()
    number = ctx.get("customer")
    icao = ctx.get("icao")
    loc = ref.locations.get(icao or "")
    site = ref.site(number, icao) if number and icao else None
    role_info = ctx.get("roles") or {}
    return {
        "customer_number": number,
        "customer_name": ctx.get("customer_name"),
        "customer_source": "phone system" if ctx.get("customer_source") == "prefill" else ("heard on the call" if number else ""),
        "ani": ctx.get("ani", ""),
        "icao": icao,
        "fbo": loc.fbo if loc else "",
        "on_file_here": site is not None if number and icao else None,
        "other_locations": [s.icao for s in ref.sites_for(number) if s.icao != icao] if number else [],
        "tail_number": ctx.get("tail_number"),
        "quote_intent": bool(ctx.get("quote_intent")),
        "roles": role_info.get("roles", {}),
        "roles_confident": role_info.get("confident"),
    }


def slot_hint(call: Call) -> dict | None:
    """What the handler still needs to pin down. Today that is the tail-number slot.

    state is one of: waiting (agent re-asked, next spelled run is the answer), incomplete (part of a tail heard),
    unconfirmed (a full tail heard but below the confidence threshold or not in the fleet), confirmed.
    """
    slot = (call.context or {}).get("tail") or {}
    value, partial = slot.get("value"), slot.get("partial")
    if slot.get("pending"):
        message = (f"Waiting for the caller to repeat the tail number. Heard so far: {partial}." if partial else
                   "Waiting for the caller to give the tail number again." if value else
                   "Waiting for the caller to give the tail number.")
        return {"state": "waiting", "label": "Tail number", "message": message}
    if partial:
        from .extract import registration_country
        state = "unconfirmed" if registration_country(partial) else "incomplete"
        prompt = call.moments.filter(kind="agent_prompt", tool_args__about="tail_number").order_by("-id").first()
        message = prompt.tool_args.get("message") if prompt else ""
        if not message:
            message = (f"Tail {partial} heard but not confirmed. Read it back to the caller." if state == "unconfirmed"
                       else f"Tail incomplete ({partial}). Ask the caller to confirm the full tail number.")
        return {"state": state, "label": "Tail number", "value": partial, "message": message}
    if value:
        return {"state": "confirmed", "label": "Tail number", "value": value, "message": f"Tail {value} confirmed."}
    return None


def _resolved(m: Moment, call: Call) -> bool:
    """A tail-number prompt is resolved once a tail was confirmed after it."""
    if m.kind != "agent_prompt" or m.tool_args.get("about") != "tail_number":
        return False
    return call.moments.filter(kind="tail_number", id__gt=m.pk).exists()


def _card_for(m: Moment, resolved: bool = False) -> dict:
    """One prompt card. Use-case results carry a sheet card (Who / Want to see / Outcome / Category / Sent to)."""
    r = m.tool_result if isinstance(m.tool_result, dict) else {}
    sheet = r.get("card") or {}
    if m.kind == "quote_request":
        if r.get("found"):
            headline = (f"Offer {r['margin_floor_gppg']:.4f} to {r['current_gppg']:.4f} GPPG "
                        f"({r['addressable_volume_gal']:,.0f} gal addressable)")
            if r.get("margin_high") and r.get("revised_offer_gppg"):
                headline += f". Margin high: try {r['revised_offer_gppg']:.4f}"
        else:
            headline = r.get("note") or "Quote lookup pending."
        say = ""
    elif m.kind == "dispute_risk":
        headline, say = r.get("prompt_agent", ""), r.get("quote_wording", "")
    elif m.kind == "sentiment_shift":
        headline, say = r.get("agent_prompt", ""), ""
    elif m.kind == "tail_number":
        headline, say = f"Tail {r.get('tail_number') or m.tool_args.get('tail_number', '')}: {r.get('note', '')}".strip(": "), ""
    else:  # agent_prompt
        headline, say = r.get("message") or m.tool_args.get("message", ""), ""
    return {
        "id": m.pk, "kind": m.kind, "at": m.start, "role": m.role, "heard": m.text,
        "headline": headline, "say": say, "status": m.tool_status,
        "who": sheet.get("who", ""), "want_to_see": sheet.get("want_to_see", ""), "outcome": sheet.get("outcome", ""),
        "category": sheet.get("category", ""), "sent_to": sheet.get("alerts", ""),
        "notified": bool(r.get("dispatcher_notified") or r.get("managers_notified")),
        "resolved": resolved,
    }


def prompt_cards(call: Call) -> list[dict]:
    """Prompt cards grouped by use case, newest first in each group; groups with nothing yet are kept (empty).

    Groups with a card come first, the group with the newest card on top, so what just fired is at the top of the
    column where the handler is looking.
    """
    moments = list(call.moments.filter(kind__in=USE_CASE_KINDS).order_by("-id"))
    groups = []
    for title, kinds in CARD_GROUPS:
        cards = [_card_for(m, _resolved(m, call)) for m in moments if m.kind in kinds]
        cards.sort(key=lambda c: c["resolved"])  # open prompts above resolved ones, each newest first
        groups.append({"title": title, "sent_to": ALERTS.get(title, "Call handler"), "cards": cards,
                       "newest": max((c["id"] for c in cards), default=0)})
    groups.sort(key=lambda g: -g["newest"])
    return groups


def transcript(call: Call) -> list[dict]:
    """Transcript turns: consecutive segments from the same speaker are one turn, so pauses don't split sentences."""
    roles = call.roles
    turns = []
    for s in call.segments.all():
        if turns and turns[-1]["speaker"] == s.speaker:
            turns[-1]["text"] = f"{turns[-1]['text']} {s.text}".strip()
            turns[-1]["id"] = s.pk
            continue
        turns.append({"id": s.pk, "start": s.start, "speaker": s.speaker, "role": roles.get(s.speaker, s.speaker) or "",
                      "text": s.text})
    return turns


def headline(call: Call) -> str:
    """One line a handler can scan after the call: who, what they wanted, where, tail, and what fired.

    Falls back to winnow's headline (first decision or action item) when no use case fired.
    """
    ctx = call.context or {}
    kinds = set(call.moments.values_list("kind", flat=True))
    parts = []
    if ctx.get("customer_name"):
        parts.append(ctx["customer_name"])
    if ctx.get("quote_intent") or "quote_request" in kinds:
        parts.append(f"quote wanted at {ctx['icao']}" if ctx.get("icao") else "quote wanted")
    elif ctx.get("icao"):
        parts.append(f"about {ctx['icao']}")
    if ctx.get("tail_number"):
        parts.append(f"tail {ctx['tail_number']}")
    if "dispute_risk" in kinds:
        parts.append("dispute risk: fees restated")
    if "sentiment_shift" in kinds:
        parts.append("caller upset, managers alerted")
    return "; ".join(parts)


def version(call: Call) -> str:
    """Changes whenever anything on the call page would: lets the live script skip redraws."""
    seg = call.segments.order_by("-id").values_list("id", flat=True).first() or 0
    mom = call.moments.order_by("-id").values_list("id", flat=True).first() or 0
    state = repr((call.pending_text, sorted(call.roles.items()), (call.context or {}).get("tail")))
    return f"{call.status}:{seg}:{mom}:{zlib.crc32(state.encode()):08x}"


def manager_alerts(live_only: bool = False, limit: int = 100) -> list[dict]:
    """Alerts that went beyond the call handler (managers / dispatcher), newest first, across calls."""
    qs = Moment.objects.filter(kind__in=ALERT_KINDS).select_related("call").order_by("-id")
    if live_only:
        qs = qs.filter(call__status="live")
    out = []
    for m in qs[:limit]:
        card = _card_for(m)
        card.update(call_id=m.call_id, call_title=str(m.call), call_status=m.call.status, alerted_at=m.call.created_at + timedelta(seconds=m.start or 0),
                    customer=(m.call.context or {}).get("customer_name") or "",
                    avg_sentiment=(m.tool_result or {}).get("avg_sentiment") if isinstance(m.tool_result, dict) else None,
                    recent=(m.tool_result or {}).get("recent", []) if isinstance(m.tool_result, dict) else [],
                    message=(m.tool_result or {}).get("dispatcher_alert", "") if isinstance(m.tool_result, dict) else "")
        out.append(card)
    return out


def test_calls() -> list[dict]:
    """WAV files in TEST_CALLS_DIR for the entry page: name, length, and whether hand-labeled turns sit next to it."""
    folder = settings.TEST_CALLS_DIR
    if not folder.is_dir():
        return []
    out = []
    for path in sorted(folder.glob("*.wav")):
        try:
            info = sf.info(str(path))
        except RuntimeError:
            continue
        truth = path.with_suffix(".truth.json")
        turns = len(json.loads(truth.read_text())) if truth.exists() else 0
        out.append({"name": path.name, "path": str(path), "duration": round(info.duration, 1),
                    "channels": info.channels, "labeled_turns": turns,
                    "scripted": path.with_suffix(".script.json").exists(),
                    "single_leg": path.stem.endswith(("_agent", "_caller"))})
    # Whole calls first (scripted demo calls on top), single legs of a two-leg call last.
    out.sort(key=lambda c: (c["single_leg"], not c["scripted"], c["name"]))
    return out
