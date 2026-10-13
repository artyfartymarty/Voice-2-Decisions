"""Static reference data the use-case tools look things up in.

Two flat files in REFERENCE_DIR (synthetic copies in fixtures/reference/; point REFERENCE_DIR at real exports at work):

    customers.csv       CUSTOMER_NAME, CUSTOMER_NUMBER, ICAO, P12M_TOTAL_WFS_VOL, P12M_ADDRESSABLE_VOL, MIN_GPPG, CURRENT_GPPG
    location_notes.csv  ICAO, IATA, Country, Region, Supplier, FBO_NUMBER, FBO, Private Notes, Public Notes (HTML)

Finding customers, locations and tail numbers in speech lives in extract.py.
"""
import csv
import html
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

DISPUTE_WORDS = re.compile(r"\b(disputes?|disputed|chargebacks?|billing complaints?)\b", re.IGNORECASE)
NOTE_BREAK = re.compile(r"\n|•|\||;|(?<!\d)\.(?!\d)")  # a period inside $1.10 is not a break
FEE_WORD = re.compile(r"\bfees?\b", re.IGNORECASE)
COUNT = re.compile(r"\b(\d+)\b(?!\s*(?:months?|hrs?|hours?|days?|years?|yrs?|gal|%))", re.IGNORECASE)


def html_to_text(raw: str) -> str:
    """Messy notes HTML -> plain lines."""
    text = re.sub(r"(?i)<\s*(br|/p|/div|/li|li)\b[^>]*>", "\n", raw or "")
    text = html.unescape(re.sub(r"<[^>]+>", "", text)).replace("\xa0", " ")
    lines = [re.sub(r"\s+", " ", line).strip(" -•|") for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


@dataclass
class Location:
    icao: str
    iata: str
    country: str
    region: str
    supplier: str
    fbo_number: str
    fbo: str
    private_notes: str
    public_notes: str

    @property
    def fees(self) -> list[str]:
        """Fee lines from the public notes (what the customer is told)."""
        seen, out = set(), []
        for piece in NOTE_BREAK.split(self.public_notes):
            line = piece.strip(" ,.")
            if FEE_WORD.search(line) and line.lower() not in seen:
                seen.add(line.lower())
                out.append(line)
        return out

    @property
    def dispute_mentions(self) -> list[str]:
        return [line for line in self.private_notes.splitlines() if DISPUTE_WORDS.search(line)]

    @property
    def dispute_count(self) -> int:
        """Best effort: the first count on each line that mentions disputes (not "12 months"), else 1 per line."""
        total = 0
        for line in self.dispute_mentions:
            found = COUNT.search(line)
            total += int(found.group(1)) if found else 1
        return total


@dataclass
class CustomerSite:
    customer_name: str
    customer_number: str
    icao: str
    total_volume: float
    addressable_volume: float
    min_gppg: float
    current_gppg: float


@dataclass
class DisputeRow:
    customer_number: str
    icao: str
    count: int
    reason: str


@dataclass
class Reference:
    sites: list[CustomerSite] = field(default_factory=list)
    locations: dict[str, Location] = field(default_factory=dict)
    disputes: list[DisputeRow] = field(default_factory=list)  # optional disputes.csv

    def dispute_profile(self, icao: str | None, customer_number: str | None) -> dict:
        """How dispute-prone this quote is: location (notes + dispute list), customer (dispute list), and ranks."""
        loc = self.locations.get(icao or "")
        by_loc: dict[str, int] = {}
        by_cust: dict[str, int] = {}
        for d in self.disputes:
            by_loc[d.icao] = by_loc.get(d.icao, 0) + d.count
            by_cust[d.customer_number] = by_cust.get(d.customer_number, 0) + d.count
        loc_count = max(loc.dispute_count if loc else 0, by_loc.get(icao or "", 0))
        cust_count = by_cust.get(customer_number or "", 0)
        rank = lambda counts, key: (sorted(counts.values(), reverse=True).index(counts[key]) + 1) if counts.get(key) else None
        pair = [d for d in self.disputes if d.icao == icao and d.customer_number == customer_number and d.count]
        return {
            "location_disputes": loc_count, "location_rank": rank(by_loc, icao or ""),
            "customer_disputes": cust_count, "customer_rank": rank(by_cust, customer_number or ""),
            "this_customer_here": pair[0].count if pair else 0, "top_reason": pair[0].reason if pair else "",
            "locations_ranked": len(by_loc), "customers_ranked": len(by_cust),
        }

    # --- lookups ---
    def customer_numbers(self) -> dict[str, str]:
        return {s.customer_number: s.customer_name for s in self.sites}

    def site(self, customer_number: str, icao: str) -> CustomerSite | None:
        return next((s for s in self.sites if s.customer_number == customer_number and s.icao == icao), None)

    def sites_for(self, customer_number: str) -> list[CustomerSite]:
        return [s for s in self.sites if s.customer_number == customer_number]


def load(directory: str | Path) -> Reference:
    directory = Path(directory)
    ref = Reference()
    with open(directory / "customers.csv", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            ref.sites.append(CustomerSite(
                customer_name=row["CUSTOMER_NAME"].strip(), customer_number=row["CUSTOMER_NUMBER"].strip(),
                icao=row["ICAO"].strip().upper(), total_volume=float(row["P12M_TOTAL_WFS_VOL"] or 0),
                addressable_volume=float(row["P12M_ADDRESSABLE_VOL"] or 0),
                min_gppg=float(row["MIN_GPPG"] or 0), current_gppg=float(row["CURRENT_GPPG"] or 0),
            ))
    with open(directory / "location_notes.csv", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            icao = row["ICAO"].strip().upper()
            ref.locations[icao] = Location(
                icao=icao, iata=row["IATA"].strip().upper(), country=row["Country"].strip(), region=row["Region"].strip(),
                supplier=row["Supplier"].strip(), fbo_number=row["FBO_NUMBER"].strip(), fbo=row["FBO"].strip(),
                private_notes=html_to_text(row["Private Notes"]), public_notes=html_to_text(row["Public Notes"]),
            )
    path = directory / "disputes.csv"
    if path.exists():
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                ref.disputes.append(DisputeRow(row["CUSTOMER_NUMBER"].strip(), row["ICAO"].strip().upper(),
                                               int(row["DISPUTES_P12M"] or 0), (row.get("TOP_REASON") or "").strip()))
    return ref


@lru_cache(maxsize=4)
def _cached(directory: str) -> Reference:
    return load(directory)


def get() -> Reference:
    from django.conf import settings  # imported here so decisions/ can use load() without Django

    return _cached(str(settings.REFERENCE_DIR))
