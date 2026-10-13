"""The functions, and the variables each one needs before it is called.

A function is called automatically as soon as every variable in `needs` is filled, and again whenever one of its
inputs (needs + optional) changes. Its result waits in the call state until the decision model surfaces it.

    function          needs                         answers
    customer_profile  customer_number               "what do you have on file for us?"
    location_info     icao                          hours, notice and fees at an airport / FBO
    fuel_quote        customer_number, icao         "what's your price per gallon there?"
    dispute_check     icao (+ customer_number)      fee complaints, surprise charges, dispute threats
    tail_check        tail_number (+ customer)      "is N88NW on our account?"
    escalate          (nothing; never auto-called)  an angry caller who needs a manager

Lookups read the flat files in REFERENCE_DIR (customers.csv, location_notes.csv, disputes.csv, fleet.csv), the same
ones the rest of the repo uses. Add a function by appending a Function to FUNCTIONS; its `description` is what the
decision model reads, so write it as "the caller ...".
"""
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Function:
    name: str
    description: str  # read by the decision model: when is this the right answer to what the caller said?
    needs: tuple[str, ...]
    run: Callable[..., dict]
    optional: tuple[str, ...] = ()
    auto: bool = True  # False: only called when the decision model picks it
    asks: dict[str, str] = field(default_factory=dict)  # variable -> what the agent should ask for when it's missing


def _hours_and_notice(loc) -> list[str]:
    keep = ("hour", "notice", "urgent", "ppr", "h24", "24/7")
    return [line for line in loc.public_notes.splitlines() if any(k in line.lower() for k in keep)][:4]


def customer_profile(ref, fleet, customer_number):
    sites = ref.sites_for(customer_number)
    name = ref.customer_numbers().get(customer_number, "")
    tails = sorted(fleet.get(customer_number, ()))
    disputes = ref.dispute_profile(None, customer_number)["customer_disputes"]
    locs = [f"{s.icao} ({ref.locations[s.icao].fbo})" if s.icao in ref.locations else s.icao for s in sites]
    return {"customer_number": customer_number, "customer": name, "locations": [s.icao for s in sites],
            "tails_on_file": tails, "disputes_p12m": disputes,
            "say": f"{name} ({customer_number}) fuels at {', '.join(locs) or 'no locations on file'}"
                   + (f"; tails on file: {', '.join(tails)}" if tails else "") + "."}


def location_info(ref, fleet, icao):
    loc = ref.locations[icao]
    info, fees = _hours_and_notice(loc), loc.fees
    return {"icao": icao, "iata": loc.iata, "fbo": loc.fbo, "country": loc.country, "info": info, "fees": fees,
            "say": f"{icao} {loc.fbo}: {'; '.join(info) or 'no hours on file'}. Fees: {'; '.join(fees) or 'none listed'}."}


def fuel_quote(ref, fleet, customer_number, icao):
    name = ref.customer_numbers().get(customer_number, "")
    site = ref.site(customer_number, icao)
    loc = ref.locations.get(icao)
    if site is None:
        on_file = [s.icao for s in ref.sites_for(customer_number)]
        return {"customer": name, "customer_number": customer_number, "icao": icao, "found": False,
                "locations_on_file": on_file,
                "say": f"No pricing history for {name} at {icao}. Quote from list price; on file: {', '.join(on_file) or 'none'}."}
    room = round(site.current_gppg - site.min_gppg, 4)
    return {"customer": name, "customer_number": customer_number, "icao": icao, "fbo": loc.fbo if loc else "",
            "found": True, "current_gppg": site.current_gppg, "floor_gppg": site.min_gppg, "room_gppg": room,
            "addressable_gal": site.addressable_volume, "total_gal": site.total_volume,
            "fees": loc.fees if loc else [],
            "say": f"Quote {name} at {icao}: current {site.current_gppg:.4f} GPPG. Don't go below the minimum, "
                   f"{site.min_gppg:.4f} GPPG ({room:.4f} of room)."}


def dispute_check(ref, fleet, icao, customer_number=None):
    loc = ref.locations[icao]
    p = ref.dispute_profile(icao, customer_number)
    fees = "; ".join(loc.fees) or "see the fee schedule"
    risky = p["location_disputes"] or p["customer_disputes"]
    reason = f" Last issue: {p['top_reason']}." if p["top_reason"] else ""
    say = (f"Dispute risk at {icao}: {p['location_disputes']} location disputes, {p['customer_disputes']} for this customer."
           f"{reason} Restate the fees and put them in writing: {fees}." if risky else
           f"No dispute history at {icao}. Fees: {fees}.")
    return {"icao": icao, "fbo": loc.fbo, "customer_number": customer_number, **p, "fees": loc.fees,
            "quote_wording": f"Fuel price excludes location fees. Fees at {loc.fbo} ({icao}) apply as published: {fees}.",
            "say": say}


def tail_check(ref, fleet, tail_number, customer_number=None):
    tails = fleet.get(customer_number or "", set())
    on_file = (tail_number in tails) if tails else None
    say = (f"{tail_number} is on file for this customer." if on_file else
           f"{tail_number} is NOT on file for this customer. Confirm it." if tails else
           f"{tail_number} is a valid registration; no fleet on file to check it against.")
    return {"tail_number": tail_number, "customer_number": customer_number, "on_file": on_file, "say": say}


def escalate(ref, fleet, customer_number=None):
    return {"customer_number": customer_number, "alert": "managers",
            "say": "Caller is frustrated. Acknowledge it, don't argue the charge on the call, offer a manager callback."}


FUNCTIONS = [
    Function("fuel_quote", "the caller wants a fuel price, a quote, a rate, or what it costs per gallon",
             ("customer_number", "icao"), fuel_quote,
             asks={"customer_number": "the account number or company name", "icao": "which airport or FBO"}),
    Function("dispute_check", "the caller questions or complains about a fee or charge they were billed, asks what fees "
             "they will be charged, or threatens a dispute", ("icao",), dispute_check, optional=("customer_number",),
             asks={"icao": "which airport the charge was at"}),
    Function("location_info", "the caller asks about an airport or FBO: opening hours, notice needed, or how to arrange fuel",
             ("icao",), location_info, asks={"icao": "which airport or FBO"}),
    Function("tail_check", "the caller gives an aircraft tail number or registration, or asks if an aircraft is on their account",
             ("tail_number",), tail_check, optional=("customer_number",), asks={"tail_number": "the full tail number"}),
    Function("customer_profile", "the caller asks what is on file for their account: their locations or aircraft",
             ("customer_number",), customer_profile, asks={"customer_number": "the account number or company name"}),
    Function("escalate", "the caller is angry or fed up, threatens to take their business elsewhere, or asks for a manager",
             (), escalate, optional=("customer_number",), auto=False),
]
BY_NAME = {f.name: f for f in FUNCTIONS}
NONE_DESCRIPTION = "small talk, a greeting, thanks, or the caller only gave details; nothing to look up"
