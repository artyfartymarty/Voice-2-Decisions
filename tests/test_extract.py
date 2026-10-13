"""Tool arguments from speech without an LLM: extractor + per-call tail-number slot."""
import time

import pytest

from copilot import extract
from copilot.winnow import Winnower


@pytest.fixture
def ex():
    return extract.get()


# --- customers ---

@pytest.mark.parametrize("text,number", [
    ("this is Dana from Northwind Aviation", "204511"),
    ("North Wind aviation calling", "204511"),            # ASR split the name
    ("account two zero four five one one", "204511"),    # spoken digits
    ("account 204,511", "204511"),
    ("account 2 0 4 5 1 1", "204511"),
    ("Blue Herron Jet Charter here", "318842"),          # misspelled by ASR
    ("Summit Ridge Flight Ops", "129907"),
    ("we fly out of the ridge", None),
])
def test_customer(ex, text, number):
    e = ex.customer(text)
    assert (e.value if e else None) == number


# --- locations ---

@pytest.mark.parametrize("text,icao", [
    ("landing at KTEB", "KTEB"),
    ("K T E B", "KTEB"),
    ("arriving at K-T-E-B", "KTEB"),
    ("kilo tango echo bravo", "KTEB"),                   # phonetic alphabet
    ("into TEB tomorrow", "KTEB"),
    ("flying into Teterboro", "KTEB"),                   # alias
    ("flying into Teeterborough", "KTEB"),               # how ASR actually wrote it
    ("at Meadowlark Jet Center", "KTEB"),                # FBO name
    ("Van Nuys please", "KVNY"),
    ("Paris Le Bourget", "LFPB"),
    ("London Luton on Friday", "EGGW"),
    ("we'll take the best rate", None),
])
def test_location(ex, text, icao):
    e = ex.location(text)
    assert (e.value if e else None) == icao


# --- tail numbers, one utterance ---

@pytest.mark.parametrize("text,tail", [
    ("tail number November five one two bravo tango", "N512BT"),
    ("the tail is N512BT", "N512BT"),
    ("registration golf charlie romeo whiskey lima", "GCRWL"),
    ("it's Charlie golf hotel bravo juliet", "CGHBJ"),
    ("November eight eight November whiskey", "N88NW"),
    ("tail number niner zero zero Sierra Romeo", None),  # no N prefix: not a complete US registration
])
def test_tail_grammar(ex, text, tail):
    e = ex.tail_number(text)
    assert (e.value if e else None) == tail


# --- the slot, across turns (Brian's cases) ---

def feed(lines, customer=None):
    """lines: (speaker, text). Returns (winnower, moments)."""
    w = Winnower(context={"customer": customer} if customer else None)
    out = []
    for speaker, text in lines:
        out += w.feed(text, final=True, speaker=speaker)
    return w, out


def tails(moments):
    return [m.tool_call["args"]["tail_number"] for m in moments if m.kind == "tail_number"]


def prompts(moments):
    return [m.tool_call["args"]["message"] for m in moments if m.kind == "agent_prompt"]


def test_cutoff_then_reask_then_repeat():
    w, out = feed([
        ("caller", "Tail number is bravo tango"),                      # cut off
        ("agent", "Sorry, what was the tail number again?"),
        ("caller", "November five one two bravo tango."),
    ])
    assert tails(out) == ["N512BT"]                                    # fires once, only for the complete one
    assert len(prompts(out)) == 1 and "incomplete" in prompts(out)[0] and "BT" in prompts(out)[0]
    assert w.context["tail"]["value"] == "N512BT" and not w.context["tail"]["pending"]


def test_cutoff_with_known_fleet_suggests_the_tail():
    _, out = feed([("caller", "Tail number is bravo tango")], customer="204511")
    assert tails(out) == []
    assert "On file for this customer: N512BT" in prompts(out)[0]


def test_correction_replaces_letters():
    w, out = feed([
        ("caller", "Tail number November five one two bravo delta."),
        ("agent", "Five one two bravo delta?"),
        ("caller", "No, that's B T, not B D."),
    ])
    assert tails(out) == ["N512BD", "N512BT"]
    corrected = [m for m in out if m.kind == "tail_number"][-1].tool_call["args"]
    assert corrected["how"] == "corrected" and corrected["replaces"] == "N512BD"
    assert w.context["tail"]["value"] == "N512BT"


def test_correction_other_order():
    _, out = feed([("caller", "Tail November five one two bravo delta."), ("caller", "Sorry, not bravo delta, bravo tango.")])
    assert tails(out)[-1] == "N512BT"


def test_spread_out_spelling_is_joined():
    _, out = feed([
        ("caller", "The tail number is November five one two"),       # valid N-number on its own, but...
        ("caller", "bravo tango"),
    ], customer="204511")
    # N512 alone isn't in the customer's fleet, so it only prompts; the joined N512BT is on file and fires.
    assert tails(out) == ["N512BT"]


def test_tail_in_fleet_needs_no_cue():
    _, out = feed([("caller", "We're bringing N88NW in.")], customer="204511")
    assert tails(out) == ["N88NW"]


def test_tail_not_in_fleet_asks_to_confirm():
    _, out = feed([("caller", "Tail number November four four four alpha alpha.")], customer="204511")
    assert tails(out) == []
    assert "isn't on file" in prompts(out)[0]


def test_dates_and_codes_are_not_tails():
    _, out = feed([("caller", "We land November three at kilo tango echo bravo, account two zero four five one one.")])
    assert tails(out) == [] and prompts(out) == []


# --- speed: the extractor must keep up with partial transcripts ---

def test_extractor_is_fast(ex):
    lines = ["Hi this is Dana with North Wind Aviation account two zero four five one one",
             "we are flying into Teeterborough on Thursday tail number November five one two bravo tango",
             "what's your price per gallon, honestly I'm frustrated about the callout fee"] * 30
    ex.extract(lines[0])  # warm caches
    times = []
    for line in lines:
        t = time.perf_counter()
        ex.extract(line)
        times.append(time.perf_counter() - t)
    p95 = sorted(times)[int(len(times) * 0.95)]
    print(f"\nextract p95 {p95 * 1000:.2f} ms over {len(times)} updates")
    assert p95 < 0.025  # generous for CI; typically a few ms


# --- through the API ---

@pytest.mark.django_db
def test_customer_prefill_lets_quote_fire_on_location_alone(client):
    import json
    call = client.post("/api/calls/live/", json.dumps({"customer_id": "204511", "ani": "+15551230000"}),
                       content_type="application/json").json()
    assert call["context"]["customer_name"] == "NORTHWIND AVIATION, LLC"
    url = f"/api/calls/{call['id']}/partial/"
    r = client.post(url, json.dumps({"text": "What's your price per gallon at Van Nuys?", "speaker": "caller"}),
                    content_type="application/json").json()
    quote = next(m for m in r["new_moments"] if m["kind"] == "quote_request")
    assert quote["tool_result"]["icao"] == "KVNY" and quote["tool_result"]["found"] is True


@pytest.mark.django_db
def test_unknown_prefill_is_ignored(client):
    import json
    call = client.post("/api/calls/live/", json.dumps({"customer_id": "999999"}), content_type="application/json").json()
    assert "customer" not in call["context"]


@pytest.mark.django_db
def test_live_tail_reask_through_api(client):
    import json
    call = client.post("/api/calls/live/", json.dumps({"customer_id": "204511"}), content_type="application/json").json()
    url = f"/api/calls/{call['id']}/partial/"
    kinds = []
    for spk, text in [("caller", "Tail number is bravo tango."), ("agent", "Sorry, what was the tail number again?"),
                      ("caller", "November five one two bravo tango.")]:
        r = client.post(url, json.dumps({"text": text, "speaker": spk}), content_type="application/json").json()
        kinds.append([(m["kind"], m["tool_result"]) for m in r["new_moments"] if m["kind"] in ("tail_number", "agent_prompt")])
    assert kinds[0][0][0] == "agent_prompt" and "N512BT" in kinds[0][0][1]["message"]
    assert kinds[1] == []
    assert kinds[2][0][0] == "tail_number" and kinds[2][0][1]["on_file"] is True
    page = client.get(f"/calls/{call['id']}/").content.decode()
    assert "Ask the caller to repeat the full tail number" in page and "Tail N512BT: On file" in page
    assert "Tail N512BT confirmed." in page  # the slot hint once the tail is complete


def test_boost_words(client):
    phrases = client.get("/api/boost-words/").json()["phrases"]
    assert {"Northwind Aviation", "Teterboro", "Meadowlark Jet Center", "KTEB"} <= set(phrases)
