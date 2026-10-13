"""The three use-case tools winnow calls (see the use-case sheet):

    quote_lookup      GP: customer + location -> addressable volume, price range within margin, alert dispatcher
    dispute_guidance  Avoiding disputes: high-dispute location -> prompt agent to restate fees, wording for the quote
    alert_manager     Customer experience: negative sentiment shift -> alert managers in real time

Alerts go through notify(): always logged, and POSTed as JSON to NOTIFY_WEBHOOK_URL when that is set
(Slack/Teams incoming webhooks, or any internal endpoint). Swap notify() to change where alerts land.
"""
import json
import logging
import urllib.request

from django.conf import settings

from . import reference
from .tools import tool

log = logging.getLogger(__name__)


def notify(audience: str, message: str, payload: dict) -> bool:
    """Send an alert to `audience` (dispatcher, managers). Returns True if a webhook accepted it."""
    log.warning("ALERT [%s] %s", audience, message)
    url = getattr(settings, "NOTIFY_WEBHOOK_URL", "")
    if not url:
        return False
    body = json.dumps({"audience": audience, "text": f"[{audience}] {message}", "payload": payload}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            return 200 <= resp.status < 300
    except Exception:
        log.exception("notify webhook failed")
        return False


def _info_lines(loc, limit=4) -> list[str]:
    keep = ("hour", "notice", "urgent", "ppr", "h24", "24/7")
    return [line for line in loc.public_notes.splitlines() if any(k in line.lower() for k in keep)][:limit]


@tool()
def quote_lookup(text, customer_number, icao, at=None):
    ref = reference.get()
    loc = ref.locations.get(icao)
    site = ref.site(customer_number, icao)
    name = ref.customer_numbers().get(customer_number, "")
    base = {"customer": name, "customer_number": customer_number, "icao": icao, "fbo": loc.fbo if loc else ""}
    if site is None:
        return {**base, "found": False,
                "note": "No P12M history for this customer at this location. Quote from list price.",
                "customer_locations": [s.icao for s in ref.sites_for(customer_number)]}
    result = {
        **base, "found": True,
        "addressable_volume_gal": site.addressable_volume,
        "total_volume_gal": site.total_volume,
        "margin_floor_gppg": site.min_gppg,
        "current_gppg": site.current_gppg,
        "offer_range_gppg": [site.min_gppg, site.current_gppg],
        "suggested_gppg": site.current_gppg,
        "location_info": _info_lines(loc) if loc else [],
    }
    if site.addressable_volume <= 0:
        result["note"] = "No addressable volume at this location. Check with pricing before offering."
    message = (f"Quote in progress: {name} ({customer_number}) at {icao}{' / ' + loc.fbo if loc else ''}. "
               f"{site.addressable_volume:,.0f} gal addressable P12M. Offer {site.min_gppg:.4f}-{site.current_gppg:.4f} GPPG.")
    result["dispatcher_alert"] = message
    result["dispatcher_notified"] = notify("dispatcher", message, result)
    return result


@tool()
def dispute_guidance(text, icao, at=None):
    loc = reference.get().locations.get(icao)
    if loc is None:
        return {"icao": icao, "found": False}
    fees = loc.fees
    fee_list = "; ".join(fees) if fees else "see location fee schedule"
    return {
        "icao": icao, "fbo": loc.fbo, "found": True,
        "dispute_count": loc.dispute_count,
        "prompt_agent": f"High-dispute location ({loc.dispute_count} disputes). Restate the fees to the customer before closing: {fee_list}.",
        "quote_wording": f"Fuel price excludes location fees. Fees at {loc.fbo} ({icao}) apply as published: {fee_list}.",
        "fees": fees,
        "internal_notes": loc.dispute_mentions,
    }


@tool()
def alert_manager(text, avg_sentiment, recent=(), speaker="", at=None):
    who = f"the {speaker}'s" if speaker else "the"
    message = f"Negative sentiment shift in {who} side of a live call (avg {avg_sentiment:+.2f}): \"{text}\""
    payload = {"speaker": speaker, "avg_sentiment": avg_sentiment, "recent": list(recent), "at": at}
    return {"alerted": True, "speaker": speaker, "avg_sentiment": avg_sentiment, "recent": list(recent),
            "managers_notified": notify("managers", message, payload)}
