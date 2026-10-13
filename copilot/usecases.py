"""The use cases from the sheet, each built the way its Pattern column says:

    Category            Pattern                                   Tool              Model?
    GP                  Static only, Customer and Location req.   quote_lookup      no: flat-file lookup on customer + ICAO
    Avoiding Disputes   Single key word, static data search       dispute_guidance  no: keyword trigger + notes + dispute list
    Customer Experience Sentiment and orchestration               alert_manager     yes: sentiment model (Winnow) + alert routing

Every result carries a `card` with the sheet's columns (category, who, want_to_see, outcome, pattern) so the UI
and API can show it the way the sheet is laid out. Supporting tools (validate_tail_number, prompt_agent) fill and
guard the arguments.

Alerts go through notify(): always logged, and POSTed as JSON to the configured webhooks (Slack/Teams incoming
webhooks or any internal endpoint). Change notify() to change where alerts land.
"""
import json
import logging
import urllib.request

from django.conf import settings

from . import reference
from .tools import tool

log = logging.getLogger(__name__)

WHO = "Customer"  # the sheet's Who column: who the use case concerns (the same for all three rows)
ALERTS = {  # who receives it (not a sheet column; kept so the UI and routing can say where it went)
    "GP": "Dispatcher",
    "Avoiding Disputes": "Call handler",
    "Customer Experience": "Call handler, managers / leads",
}
PATTERN = {
    "GP": "Static only, Customer and Location required",
    "Avoiding Disputes": "Single key word, static data search",
    "Customer Experience": "Sentiment and orchestration",
}


def card(category: str, want_to_see: str, outcome: str) -> dict:
    return {"category": category, "who": WHO, "want_to_see": want_to_see, "outcome": outcome,
            "pattern": PATTERN[category], "alerts": ALERTS[category]}


def _post(url: str, body: dict) -> bool:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            return 200 <= resp.status < 300
    except Exception:
        log.exception("notify webhook %s failed", url)
        return False


def notify(audience: str, message: str, payload: dict) -> bool:
    """Send an alert to `audience` (dispatcher, managers, yield). Returns True if any webhook accepted it."""
    log.warning("ALERT [%s] %s", audience, message)
    urls = list(settings.MANAGER_WEBHOOKS) if audience == "managers" and settings.MANAGER_WEBHOOKS else []
    if not urls and settings.NOTIFY_WEBHOOK_URL:
        urls = [settings.NOTIFY_WEBHOOK_URL]
    body = {"audience": audience, "text": f"[{audience}] {message}", "payload": payload}
    return any([_post(u, body) for u in urls]) if urls else False


def _info_lines(loc, limit=4) -> list[str]:
    keep = ("hour", "notice", "urgent", "ppr", "h24", "24/7")
    return [line for line in loc.public_notes.splitlines() if any(k in line.lower() for k in keep)][:limit]


# --- GP ---------------------------------------------------------------------------------------------------

@tool()
def quote_lookup(text, customer_number, icao, at=None):
    """Customer called for a quote: addressable volume at the quoted location, dispatcher alert with the margin
    reduction range; checks addressable volume (feedback to Yield) and whether the margin is high."""
    ref = reference.get()
    loc = ref.locations.get(icao)
    site = ref.site(customer_number, icao)
    name = ref.customer_numbers().get(customer_number, "")
    operates_at = [s.icao for s in ref.sites_for(customer_number)]
    base = {"customer": name, "customer_number": customer_number, "icao": icao, "fbo": loc.fbo if loc else "",
            "customer_locations": operates_at}
    if site is None:
        result = {**base, "found": False,
                  "note": "No P12M history for this customer at this location. Quote from list price and tell Yield."}
        result["yield_feedback"] = f"For Yield: {name} ({customer_number}) asked for a quote at {icao}, which has no row in the flat file."
        result["yield_notified"] = notify("yield", result["yield_feedback"], base)
        result["card"] = card("GP", f"{name} at {icao}: no volume on file. Locations on file: {', '.join(operates_at) or 'none'}.",
                              "Volume at this location may be missing; fed back to Yield.")
        return result

    reduction = round(site.current_gppg - site.min_gppg, 4)
    margin_high = site.min_gppg > 0 and site.current_gppg >= settings.MARGIN_HIGH_RATIO * site.min_gppg
    revised = round(site.min_gppg + reduction / 2, 4)
    ratio = site.addressable_volume / site.total_volume if site.total_volume else 0.0
    volume_check = ("no addressable volume" if site.addressable_volume <= 0 else
                    "addressable exceeds total" if site.addressable_volume > site.total_volume else
                    f"only {ratio:.0%} of total is addressable" if ratio < settings.ADDRESSABLE_CHECK_RATIO else "")
    result = {
        **base, "found": True,
        "addressable_volume_gal": site.addressable_volume,
        "total_volume_gal": site.total_volume,
        "margin_floor_gppg": site.min_gppg,
        "current_gppg": site.current_gppg,
        "offer_range_gppg": [site.min_gppg, site.current_gppg],
        "margin_reduction_range_gppg": [0.0, reduction],
        "margin_high": margin_high,
        "revised_offer_gppg": revised if margin_high else None,
        "volume_check": volume_check or "ok",
        "location_info": _info_lines(loc) if loc else [],
    }
    message = (f"{name} ({customer_number}) is asking for a quote at {icao}{' / ' + loc.fbo if loc else ''}. "
               f"{site.addressable_volume:,.0f} gal addressable P12M. Current {site.current_gppg:.4f} GPPG; "
               f"you can come down by up to {reduction:.4f} (floor {site.min_gppg:.4f}). "
               f"Any feedback on the quoted price or more info from the customer?")
    if margin_high:
        message += f" Margin looks high: consider {revised:.4f} GPPG."
    result["dispatcher_alert"] = message
    result["dispatcher_notified"] = notify("dispatcher", message, result)
    outcomes = []
    if volume_check:
        result["yield_feedback"] = f"For Yield: check addressable volume for {name} at {icao} ({volume_check})."
        result["yield_notified"] = notify("yield", result["yield_feedback"], base)
        outcomes.append("Addressable volume flagged and fed back to Yield")
    else:
        outcomes.append("Addressable volume looks consistent")
    outcomes.append(f"Margin high: offer revised price {revised:.4f} GPPG (range {site.min_gppg:.4f} to {site.current_gppg:.4f})"
                    if margin_high else f"Margin within range: {site.min_gppg:.4f} to {site.current_gppg:.4f} GPPG")
    result["card"] = card(
        "GP",
        f"{name} at {icao}: {site.addressable_volume:,.0f} gal addressable of {site.total_volume:,.0f} P12M. "
        f"Margin reduction range up to {reduction:.4f} GPPG. Dispatcher alerted.",
        ". ".join(outcomes) + ".")
    return result


# --- Avoiding Disputes ------------------------------------------------------------------------------------

@tool()
def dispute_guidance(text, icao, customer_number=None, keyword="", at=None):
    """Keyword (quote / fee ...) with a dispute-prone location or customer: reinforce fees verbally and in writing."""
    ref = reference.get()
    loc = ref.locations.get(icao)
    if loc is None:
        return {"icao": icao, "found": False}
    profile = ref.dispute_profile(icao, customer_number)
    fees = loc.fees
    fee_list = "; ".join(fees) if fees else "see location fee schedule"
    who = []
    if profile["location_disputes"]:
        who.append(f"{icao} has {profile['location_disputes']} disputes"
                   + (f" (#{profile['location_rank']} of {profile['locations_ranked']} locations)" if profile["location_rank"] else ""))
    if profile["customer_disputes"]:
        who.append(f"this customer has {profile['customer_disputes']}"
                   + (f" (#{profile['customer_rank']} of {profile['customers_ranked']} customers)" if profile["customer_rank"] else ""))
    reason = f" Last issue here: {profile['top_reason']}." if profile["top_reason"] else ""
    prompt = f"Dispute risk: {', '.join(who)}.{reason} Restate the fees before closing: {fee_list}."
    wording = f"Fuel price excludes location fees. Fees at {loc.fbo} ({icao}) apply as published: {fee_list}."
    return {
        "icao": icao, "fbo": loc.fbo, "found": True, "keyword": keyword,
        "dispute_count": profile["location_disputes"], **profile,
        "prompt_agent": prompt, "quote_wording": wording, "fees": fees, "internal_notes": loc.dispute_mentions,
        "card": card("Avoiding Disputes", f"Reinforce fees on this quote: {fee_list}. Add the fee wording to the written quote.",
                     "Fees reinforced verbally and in writing to get ahead of a credit or rebill."),
    }


# --- Customer Experience ----------------------------------------------------------------------------------

@tool()
def alert_manager(text, avg_sentiment, recent=(), speaker="", scored_by="", at=None):
    """Orchestration: tell the agent on the call, and alert the designated managers / leads in real time."""
    side = speaker or "a speaker"
    agent_prompt = (f"Heads up: the {side}'s sentiment is turning negative. Slow down, acknowledge the issue, "
                    "and offer a next step." if speaker != "agent" else
                    "Heads up: your tone is reading as frustrated. Take a breath and reset the conversation.")
    message = f"Negative sentiment shift on a live call ({side}, avg {avg_sentiment:+.2f}): \"{text}\""
    payload = {"speaker": speaker, "avg_sentiment": avg_sentiment, "recent": list(recent), "at": at, "scored_by": scored_by}
    return {"alerted": True, "speaker": speaker, "avg_sentiment": avg_sentiment, "recent": list(recent),
            "scored_by": scored_by, "agent_prompt": agent_prompt, "managers_notified": notify("managers", message, payload),
            "card": card("Customer Experience", f"The {side}'s sentiment is shifting negative ({avg_sentiment:+.2f}).",
                         "Agent alerted on the call; designated managers / leads alerted in real time.")}


# --- argument helpers -------------------------------------------------------------------------------------

@tool()
def validate_tail_number(text, tail_number, country="", confidence=0.0, heard="", customer_number=None, how="heard",
                         replaces=None, at=None):
    """Format, registry country and (when fleet.csv has the customer) whether the customer operates this tail."""
    from . import extract

    fleet = extract.get().fleet.get(customer_number or "")
    on_file = (tail_number in fleet) if fleet else None
    note = ("On file for this customer." if on_file else "Not on file for this customer. Confirm it." if fleet
            else "Format valid. No fleet on file for this customer, so ownership isn't checked.")
    return {"tail_number": tail_number, "country": country, "valid_format": True, "confidence": confidence,
            "heard": heard, "how": how, "replaces": replaces, "customer_number": customer_number, "on_file": on_file,
            "note": note}


@tool()
def prompt_agent(text, message, about="", suggestion=None, at=None):
    """Something the agent should confirm on the call; shown on the call page and returned by the API."""
    return {"message": message, "about": about, "suggestion": suggestion}
