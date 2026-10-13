"""Static reference data the use-case tools look things up in.

Two flat files in REFERENCE_DIR (synthetic copies in fixtures/reference/; point REFERENCE_DIR at real exports at work):

    customers.csv       CUSTOMER_NAME, CUSTOMER_NUMBER, ICAO, P12M_TOTAL_WFS_VOL, P12M_ADDRESSABLE_VOL, MIN_GPPG, CURRENT_GPPG
    location_notes.csv  ICAO, IATA, Country, Region, Supplier, FBO_NUMBER, FBO, Private Notes, Public Notes (HTML)

Also finds customers and locations mentioned in transcript text, allowing for how ASR writes them
("Northwind Aviation" for "NORTHWIND AVIATION, LLC", "K T E B" or "TEB" for KTEB).
"""
import csv
import html
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from django.conf import settings

SUFFIXES = re.compile(r"\b(inc|llc|ltd|corp|co|company|corporation|limited|plc|gmbh|sa)\b\.?", re.IGNORECASE)
DISPUTE_WORDS = re.compile(r"\b(disputes?|disputed|chargebacks?|billing complaints?)\b", re.IGNORECASE)
NOTE_BREAK = re.compile(r"\n|•|\||;|(?<!\d)\.(?!\d)")  # a period inside $1.10 is not a break
FEE_WORD = re.compile(r"\bfees?\b", re.IGNORECASE)
COUNT = re.compile(r"\b(\d+)\b(?!\s*(?:months?|hrs?|hours?|days?|years?|yrs?|gal|%))", re.IGNORECASE)


def squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


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
class Reference:
    sites: list[CustomerSite] = field(default_factory=list)
    locations: dict[str, Location] = field(default_factory=dict)

    # --- lookups ---
    def customer_numbers(self) -> dict[str, str]:
        return {s.customer_number: s.customer_name for s in self.sites}

    def site(self, customer_number: str, icao: str) -> CustomerSite | None:
        return next((s for s in self.sites if s.customer_number == customer_number and s.icao == icao), None)

    def sites_for(self, customer_number: str) -> list[CustomerSite]:
        return [s for s in self.sites if s.customer_number == customer_number]

    # --- finding mentions in transcript text ---
    def find_customer(self, text: str) -> tuple[str, str] | None:
        """(customer_number, customer_name) mentioned in text, by number or by name."""
        digits = re.sub(r"[^0-9 ]", "", text)
        squashed = squash(text)
        for number, name in self.customer_numbers().items():
            if re.search(rf"\b{number}\b", digits) or number in squashed:
                return number, name
        best = None
        for number, name in self.customer_numbers().items():
            core = squash(SUFFIXES.sub("", name))
            if len(core) >= 5 and core in squashed and (best is None or len(core) > best[2]):
                best = (number, name, len(core))
        return (best[0], best[1]) if best else None

    def find_location(self, text: str) -> str | None:
        """ICAO code of a location mentioned by ICAO (also spelled out), IATA, or FBO name."""
        spelled = re.sub(r"\b([A-Za-z])[\s.-](?=[A-Za-z]\b)", r"\1", text)  # "K T E B" / "K-T-E-B" -> "KTEB"
        tokens = {t.upper() for t in re.findall(r"\b[A-Za-z]{3,4}\b", spelled)}
        upper_tokens = set(re.findall(r"\b[A-Z]{3}\b", text))  # IATA only when written in capitals
        squashed = squash(text)
        for icao, loc in self.locations.items():
            if icao in tokens or (loc.iata and loc.iata in upper_tokens):
                return icao
            fbo_core = squash(re.sub(r"\b(fbo|aviation|services|center|centre|executive|handling)\b", "", loc.fbo, flags=re.IGNORECASE))
            if len(fbo_core) >= 6 and fbo_core in squashed:
                return icao
        return None


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
    return ref


@lru_cache(maxsize=4)
def _cached(directory: str) -> Reference:
    return load(directory)


def get() -> Reference:
    return _cached(str(settings.REFERENCE_DIR))
