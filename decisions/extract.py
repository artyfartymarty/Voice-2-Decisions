"""Deterministic entity extraction: tool arguments from speech, without an LLM.

The decider (rules today, Winnow later) only chooses *which* tool fires. The arguments come from here, and only
values that exist in the reference data (or match a strict grammar) can come out, so nothing is invented.

    customers      account number (digits, or spoken "two zero four five one one") or name, fuzzy + phonetic
                   against customers.csv ("North Wind Aviation" -> NORTHWIND AVIATION, LLC)
    locations      ICAO / IATA as written, spelled ("K T E B") or in the phonetic alphabet ("kilo tango echo bravo"),
                   FBO names from location_notes.csv, airport/city names from airport_aliases.csv, fuzzy
                   ("Teeterborough" -> KTEB)
    tail numbers   registration grammar with phonetic alphabet and spoken digits ("November one two three alpha
                   bravo" -> N123AB), per-country prefixes, confidence score (tools fire at >= TAIL_MIN_CONFIDENCE)

Everything works on a short window of recent words, so each partial transcript update is cheap.
"""
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache

NATO = {
    "alpha": "A", "alfa": "A", "bravo": "B", "charlie": "C", "delta": "D", "echo": "E", "foxtrot": "F", "fox": "F",
    "golf": "G", "hotel": "H", "india": "I", "juliet": "J", "juliett": "J", "kilo": "K", "lima": "L", "mike": "M",
    "november": "N", "oscar": "O", "papa": "P", "quebec": "Q", "romeo": "R", "sierra": "S", "tango": "T",
    "uniform": "U", "victor": "V", "whiskey": "W", "whisky": "W", "xray": "X", "x-ray": "X", "yankee": "Y", "zulu": "Z",
}
DIGITS = {"zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "to": "2", "too": "2", "three": "3", "tree": "3",
          "four": "4", "for": "4", "fower": "4", "five": "5", "fife": "5", "six": "6", "seven": "7", "eight": "8",
          "ait": "8", "nine": "9", "niner": "9"}
# Numbers said as numbers rather than digit by digit: "one twenty three" -> 1 2 3, "five hundred" -> 5 0 0.
TEENS = {w: str(10 + i) for i, w in enumerate(
    "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
TENS = {w: str(i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
# Registration prefixes: country -> (prefix, body pattern). US N-numbers: N + 1-5 chars, digit first, max 2 trailing letters.
REGISTRATIONS = [
    ("US", re.compile(r"^N[1-9][0-9]{0,4}$|^N[1-9][0-9]{0,3}[A-HJ-NP-Z]$|^N[1-9][0-9]{0,2}[A-HJ-NP-Z]{2}$")),
    ("CA", re.compile(r"^C[FGI][A-Z]{3}$")),
    ("UK", re.compile(r"^G[A-Z]{4}$")),
    ("DE", re.compile(r"^D[A-Z]{4}$")),
    ("FR", re.compile(r"^F[A-Z]{4}$")),
    ("CH", re.compile(r"^HB[A-Z]{3}$")),
    ("AU", re.compile(r"^VH[A-Z]{3}$")),
    ("MX", re.compile(r"^X[ABC][A-Z]{3}$")),
    ("BR", re.compile(r"^P[PRST][A-Z]{3}$")),
]
TAIL_CUE = re.compile(r"\b(tail( number)?|registration|reg|aircraft|plane|jet|november)\b", re.IGNORECASE)
SUFFIXES = re.compile(r"\b(inc|llc|ltd|corp|co|company|corporation|limited|plc|gmbh|sa)\b\.?", re.IGNORECASE)


@dataclass
class Entity:
    kind: str          # customer | location | tail_number
    value: str         # customer number, ICAO, or normalized registration
    label: str         # human-readable: customer name, FBO, country
    confidence: float
    evidence: str      # the words it came from


def squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


@lru_cache(maxsize=4096)
def phonetic(word: str) -> str:
    """A small Soundex-style key so ASR spellings of one name meet ("Teeterborough" ~ "Teterboro")."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return ""
    for a, b in (("ph", "f"), ("ck", "k"), ("gh", ""), ("ough", "o"), ("orough", "oro"), ("wh", "w"), ("kn", "n")):
        w = w.replace(a, b)
    groups = {**dict.fromkeys("bfpv", "1"), **dict.fromkeys("cgjkqsxz", "2"), **dict.fromkeys("dt", "3"), "l": "4",
              **dict.fromkeys("mn", "5"), "r": "6"}
    out, last = [w[0]], groups.get(w[0], "")
    for ch in w[1:]:
        code = groups.get(ch, "")
        if code and code != last:
            out.append(code)
        last = code
    return "".join(out)[:8]


def similarity(a: str, b: str) -> float:
    """Best of spelling similarity and phonetic match on the squashed strings."""
    sa, sb = squash(a), squash(b)
    if not sa or not sb or abs(len(sa) - len(sb)) > 0.4 * max(len(sa), len(sb)):
        return 0.0
    m = SequenceMatcher(None, sa, sb)
    if m.real_quick_ratio() < 0.75 or m.quick_ratio() < 0.75:  # cheap upper bounds first
        return 0.9 if phonetic(sa) == phonetic(sb) and min(len(sa), len(sb)) >= 5 else 0.0
    spelled = m.ratio()
    sound = 1.0 if phonetic(sa) == phonetic(sb) and min(len(sa), len(sb)) >= 5 else 0.0
    return max(spelled, 0.9 * sound)


GENERIC_WORDS = {"aviation", "air", "jet", "jets", "flight", "charter", "llc", "inc", "the", "from", "with"}


def sound_alike(a: str, b: str) -> float:
    """How alike two squashed strings look or sound: spelling ratio, or 0.9 x the ratio of their phonetic keys."""
    spelled = SequenceMatcher(None, a, b).ratio()
    ka, kb = phonetic(a), phonetic(b)
    sound = 0.9 * SequenceMatcher(None, ka, kb).ratio() if len(ka) >= 2 and len(kb) >= 2 else 0.0
    return max(spelled, sound)


def edits(a: str, b: str) -> int:
    """Characters changed, added, dropped or swapped to turn a into b (Damerau-Levenshtein): "N47JA" -> "N47AJ" is 1."""
    d = [[i + j if not i * j else 0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[-1][-1]


def words(text: str) -> list[str]:
    text = re.sub(r"\b([A-Za-z0-9])-(?=[A-Za-z0-9]\b)", r"\1 ", text)  # "K-T-E-B" -> "K T E B"; keeps "G-ABCD"
    return re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text)


def spoken_to_symbols(tokens: list[str]) -> list[str]:
    """Map each token to a symbol string: 'november'->'N', 'one'->'1', 'N123AB'->'N123AB', other words -> ''."""
    out = []
    for i, t in enumerate(tokens):
        low = t.lower()
        nxt = tokens[i + 1].lower() if i + 1 < len(tokens) else ""
        if low in TEENS:
            out.append(TEENS[low])
        elif low in TENS:  # "twenty three" -> 2 then 3; "twenty" alone -> 20
            out.append(TENS[low] + ("" if nxt in DIGITS and DIGITS[nxt] != "0" and nxt not in {"to", "too", "for", "o", "oh"} else "0"))
        elif low == "hundred" and out and out[-1].isdigit():
            out.append("00")
        elif low in NATO:
            out.append(NATO[low])
        elif low in DIGITS and low not in {"to", "too", "for"}:  # homophones of 2 and 4 are too common in speech
            out.append(DIGITS[low])
        elif re.fullmatch(r"[A-Za-z]|\d+", t):
            out.append(t.upper())
        elif (re.fullmatch(r"[A-Z0-9]{2,7}", t) and re.search(r"\d", t)) or re.fullmatch(r"[A-Z]{1,2}-?[A-Z0-9]{3,5}", t):
            out.append(t.replace("-", "").upper())  # written registrations: N123AB, G-ABCD
        else:
            out.append("")
    return out


def registration_country(reg: str) -> str:
    """Country code if `reg` is a complete registration, else ''."""
    return next((country for country, pattern in REGISTRATIONS if pattern.match(reg)), "")


@dataclass
class Run:
    """A stretch of consecutive letter/digit tokens: one attempt at spelling something out."""
    symbols: str      # "BT", "N512BT"
    heard: str        # the words: "bravo tango"
    phonetic: bool    # someone is spelling: phonetic alphabet, spoken digits, or single letters
    negated: bool     # preceded by "not" ("not B D")


def symbol_runs(text: str) -> list[Run]:
    """Runs never cross punctuation: "not bravo delta, bravo tango" is two runs."""
    runs = []
    for piece in re.split(r"[,;.!?]+", text):
        runs += _runs(piece)
    return runs


def _runs(text: str) -> list[Run]:
    toks = words(text)
    syms = spoken_to_symbols(toks)
    runs, i = [], 0
    while i < len(toks):
        if not syms[i]:
            i += 1
            continue
        j = i
        while j < len(toks) and syms[j]:
            j += 1
        span = list(range(i, j))
        # plain English "a" / "I" at the edges are words, not letters
        while span and toks[span[0]] in ("a", "A", "I") and toks[span[0]].lower() not in NATO:
            span.pop(0)
        while span and toks[span[-1]] in ("a", "A", "I"):
            span.pop()
        if span:
            sym = "".join(syms[k] for k in span)
            # spelling: phonetic alphabet, spoken digits, or two or more single letters ("B T")
            phon = (any(toks[k].lower() in NATO or toks[k].lower() in DIGITS or toks[k].lower() in TEENS
                        or toks[k].lower() in TENS for k in span)
                    or sum(len(toks[k]) == 1 and toks[k].isalpha() for k in span) >= 2)
            neg = span[0] > 0 and toks[span[0] - 1].lower() in ("not", "isn't", "wasn't")
            runs.append(Run(sym, " ".join(toks[k] for k in span), phon, neg))
        i = j
    return runs


def digit_runs(text: str, min_len: int = 4) -> list[str]:
    """Numbers said digit by digit ("two zero four five one one") or written (204511, 204,511, 2 0 4 5 1 1)."""
    runs, cur = [], ""
    for t in words(text.replace(",", "")):
        low = t.lower()
        if low in DIGITS and low not in {"to", "too", "for", "o"}:
            cur += DIGITS[low]
        elif t.isdigit():
            cur += t
        else:
            if len(cur) >= min_len:
                runs.append(cur)
            cur = ""
    if len(cur) >= min_len:
        runs.append(cur)
    return runs


class Extractor:
    """Gazetteers built once from the reference data; extract() per transcript update."""

    def __init__(self, ref, aliases: dict[str, list[str]] | None = None, tail_min_confidence: float = 0.8,
                 fleet: dict[str, set[str]] | None = None):
        self.ref = ref
        self.tail_min_confidence = tail_min_confidence
        self.fleet = fleet or {}  # customer number -> known tail numbers (normalized, no dashes)
        self.customers = {}  # number -> (name, core name)
        for number, name in ref.customer_numbers().items():
            self.customers[number] = (name, SUFFIXES.sub("", name).replace(",", " ").strip())
        firsts: dict[str, list[str]] = {}
        for number, (_, core) in self.customers.items():
            firsts.setdefault(squash(core.split()[0]) if core.split() else "", []).append(number)
        generic = {"aviation", "air", "jet", "flight", "aero", "global", "executive", "charter"}
        self.short_names = {w: nums[0] for w, nums in firsts.items() if w and len(nums) == 1 and w not in generic}
        # Initials callers use for a company: every word ("BHJC" for Blue Heron Jet Charter) or the non-generic ones (BH).
        self.initials: dict[str, list[str]] = {}
        for number, (_, core) in self.customers.items():
            parts = [w for w in re.split(r"[\s&/-]+", core) if w]
            for ws in (parts, [w for w in parts if w.lower() not in generic]):
                key = "".join(w[0] for w in ws).upper()
                if len(key) >= 2 and number not in self.initials.setdefault(key, []):
                    self.initials[key].append(number)
        self.place_names = []  # (icao, alias)
        for icao, loc in ref.locations.items():
            if loc.fbo:
                self.place_names.append((icao, loc.fbo))
            for alias in (aliases or {}).get(icao, []):
                self.place_names.append((icao, alias))
        self.icaos = set(ref.locations)
        self.iata = {loc.iata: icao for icao, loc in ref.locations.items() if loc.iata}

    # --- customers ---
    def customer(self, text: str) -> Entity | None:
        for run in digit_runs(text):
            for number, (name, _) in self.customers.items():
                if number in run:
                    return Entity("customer", number, name, 0.98, run)
        toks = words(text)
        # Short names: callers say "Northwind" for NORTHWIND AVIATION, LLC. Only when that first word is unique.
        for tok in toks:
            hit = self.short_names.get(squash(tok))
            if hit and len(squash(tok)) >= 6:
                return Entity("customer", hit, self.customers[hit][0], 0.85, tok)
        best = None
        for number, (name, core) in self.customers.items():
            n = len(core.split())
            for size in {max(1, n - 1), n, n + 1}:  # ASR may split or merge words ("North Wind" / "Northwind")
                for i in range(0, max(0, len(toks) - size) + 1):
                    window = " ".join(toks[i:i + size])
                    score = similarity(window, core)
                    if score >= 0.82 and (best is None or score > best.confidence):
                        best = Entity("customer", number, name, round(score, 3), window)
        return best

    def exact_customer(self, said: str) -> str | None:
        """The customer `said` names outright: the full name, or a first word only one customer has ("Arrow")."""
        heard = squash(said)
        for number, (_, core) in self.customers.items():
            if heard == squash(core):
                return number
        toks = words(said)
        return self.short_names.get(squash(toks[0])) if toks else None

    def customers_by_initials(self, said: str) -> list[str]:
        """Customers whose initials are `said` ("BHJC", "B.H.J.C.", "b h"); empty if it isn't initials."""
        if not re.fullmatch(r"(?:[A-Za-z][.\s]*){2,6}", said.strip()) or (" " not in said.strip() and "." not in said
                                                                          and not said.strip().isupper()):
            return []
        return self.initials.get(re.sub(r"[^A-Za-z]", "", said).upper(), [])

    def closest_customers(self, said: str, top: int = 4, floor: float = 0.7) -> list[tuple[str, str, float]]:
        """Customers whose name looks or sounds like `said` ("Arigo" -> Arrigo, Arrow, Aero ...), best first, as
        (number, name, score). The candidates Winnow chooses between when a name has no exact match."""
        heard = [w for w in words(said) if w.lower() not in GENERIC_WORDS] or words(said)
        sources = {squash(said), squash(heard[0]) if heard else ""} - {""}
        scored = []
        for number, (name, core) in self.customers.items():
            parts = core.split()
            targets = {squash(core), squash(parts[0]) if parts else "", squash(" ".join(parts[:2]))} - {""}
            score = max((sound_alike(a, b) for a in sources for b in targets), default=0.0)
            if score >= floor:
                scored.append((number, name, round(score, 3)))
        return sorted(scored, key=lambda t: -t[2])[:top]

    # --- locations ---
    def location(self, text: str) -> Entity | None:
        toks = words(text)
        symbols = spoken_to_symbols(toks)
        # codes as written ("KTEB", "TEB") and spelled or phonetic letters ("K T E B", "kilo tango echo bravo")
        for i in range(len(toks)):
            if toks[i].upper() in self.icaos and (toks[i].isupper() or len(toks[i]) == 4):
                return Entity("location", toks[i].upper(), self.ref.locations[toks[i].upper()].fbo, 0.97, toks[i])
            if toks[i] in self.iata:
                icao = self.iata[toks[i]]
                return Entity("location", icao, self.ref.locations[icao].fbo, 0.9, toks[i])
            for n in (4, 3):
                letters = symbols[i:i + n]
                if len(letters) == n and all(len(s) == 1 and s.isalpha() for s in letters):
                    code = "".join(letters)
                    if n == 4 and code in self.icaos:
                        return Entity("location", code, self.ref.locations[code].fbo, 0.95, " ".join(toks[i:i + n]))
                    if n == 3 and code in self.iata:
                        icao = self.iata[code]
                        return Entity("location", icao, self.ref.locations[icao].fbo, 0.85, " ".join(toks[i:i + n]))
        # names: FBO and airport/city aliases, fuzzy and phonetic
        best = None
        for icao, name in self.place_names:
            n = len(name.split())
            for size in {max(1, n - 1), n}:
                for i in range(0, max(0, len(toks) - size) + 1):
                    window = " ".join(toks[i:i + size])
                    if len(squash(window)) < 5:
                        continue
                    score = similarity(window, name)
                    if score >= 0.8 and (best is None or score > best.confidence):
                        best = Entity("location", icao, self.ref.locations[icao].fbo, round(score, 3), window)
        return best

    # --- tail numbers ---
    def tail_number(self, text: str) -> Entity | None:
        toks = words(text)
        symbols = spoken_to_symbols(toks)
        cue = bool(TAIL_CUE.search(text))
        best = None
        for i in range(len(toks)):
            if not symbols[i]:
                continue
            reg = ""
            for j in range(i, min(len(toks), i + 8)):
                if not symbols[j]:
                    break
                reg += symbols[j]
                if len(reg) > 7:
                    break
                for country, pattern in REGISTRATIONS:
                    if pattern.match(reg):
                        spoken = any(t.lower() in NATO or t.lower() in DIGITS for t in toks[i:j + 1])
                        conf = 0.6 + (0.25 if cue else 0) + (0.1 if spoken or len(reg) >= 5 else 0) + (0.05 if j > i else 0)
                        cand = Entity("tail_number", reg, country, round(min(conf, 0.99), 2), " ".join(toks[i:j + 1]))
                        if best is None or (len(cand.value), cand.confidence) > (len(best.value), best.confidence):
                            best = cand
        return best

    def extract(self, text: str) -> dict[str, Entity]:
        found = {}
        for kind, fn in (("customer", self.customer), ("location", self.location), ("tail_number", self.tail_number)):
            e = fn(text)
            if e:
                found[kind] = e
        return found


@lru_cache(maxsize=2)
def _extractor(directory: str, tail_min: float) -> Extractor:
    import csv
    from pathlib import Path

    from . import reference

    aliases: dict[str, list[str]] = {}
    path = Path(directory) / "airport_aliases.csv"
    if path.exists():
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                aliases.setdefault(row["ICAO"].strip().upper(), []).append(row["ALIAS"].strip())
    fleet: dict[str, set[str]] = {}
    path = Path(directory) / "fleet.csv"
    if path.exists():
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                fleet.setdefault(row["CUSTOMER_NUMBER"].strip(), set()).add(row["TAIL_NUMBER"].strip().upper().replace("-", ""))
    return Extractor(reference.load(directory), aliases, tail_min, fleet)
