"""Each use case follows its Pattern column: GP and Avoiding Disputes are static lookups, only Customer
Experience uses a model (sentiment) plus orchestration (agent + manager alerts)."""
import io
import json

import pytest

from copilot import sentiment, tools
from copilot.sentiment import LexiconSentiment, WinnowSentiment, expected_value
from copilot.winnow import Winnower, summarize


def run(lines, scorer=None, **kw):
    w = Winnower(scorer=scorer, dispute_min_count=3, **kw)
    out = []
    for speaker, text in lines:
        out += w.feed(text, final=True, speaker=speaker)
    return w, out


def kinds(out):
    return [m.kind for m in out if m.kind in ("quote_request", "dispute_risk", "sentiment_shift")]


def cards(out):
    dicts = []
    for m in out:
        d = m.as_dict()
        d["tool_result"] = tools.dispatch(m.tool_call)["result"]
        dicts.append(d)
    return summarize(dicts)["use_cases"]


# --- GP: static only, customer and location required ---

def test_gp_needs_both_slots():
    assert kinds(run([("caller", "I need a fuel quote at KTEB.")])[1]) == ["dispute_risk"]  # no customer: no GP
    assert kinds(run([("caller", "This is Northwind Aviation, I need a quote.")])[1]) == []  # no location: no GP
    _, out = run([("caller", "This is Summit Ridge Flight Ops."), ("caller", "I need a quote at Van Nuys.")])
    assert kinds(out) == ["quote_request"]  # both slots, clean customer and location: GP only


def test_gp_card_and_outcomes():
    _, out = run([("caller", "Northwind Aviation here, what's your price at KTEB?")])
    gp = next(m for m in out if m.kind == "quote_request")
    r = tools.dispatch(gp.tool_call)["result"]
    assert r["addressable_volume_gal"] == 39500 and r["customer_locations"] == ["KTEB", "KVNY", "EGGW"]
    assert r["margin_reduction_range_gppg"] == [0.0, 0.2275]
    assert r["margin_high"] is True and r["revised_offer_gppg"] == 0.1988      # 0.3125 >= 3 x 0.085
    assert "feedback on the quoted price" in r["dispatcher_alert"]
    c = r["card"]
    assert (c["category"], c["who"], c["pattern"]) == ("GP", "Dispatcher", "Static only, Customer and Location required")
    assert "39,500 gal addressable" in c["want_to_see"] and "revised price 0.1988" in c["outcome"]


def test_gp_volume_check_feeds_yield():
    _, out = run([("caller", "Summit Ridge Flight Ops, need a quote for Van Nuys.")])  # 2,500 of 4,020 addressable
    r = tools.dispatch(next(m for m in out if m.kind == "quote_request").tool_call)["result"]
    assert r["volume_check"] == "ok"
    _, out = run([("caller", "Meridian Air Partners, quote for Dallas Love Field please.")])  # 0 addressable
    r = tools.dispatch(next(m for m in out if m.kind == "quote_request").tool_call)["result"]
    assert r["volume_check"] == "no addressable volume" and "Yield" in r["yield_feedback"]


# --- Avoiding Disputes: single keyword, static data search ---

def test_dispute_fires_on_fee_keyword_without_a_quote():
    assert kinds(run([("caller", "What's the ramp fee at KTEB?")])[1]) == ["dispute_risk"]


def test_dispute_needs_a_keyword():
    assert kinds(run([("caller", "We land at KTEB on Thursday.")])[1]) == []


def test_dispute_from_the_customer_side():
    # KVNY is clean, but Blue Heron is one of the most disputing customers (5 in the dispute list)
    _, out = run([("caller", "Blue Heron Jet Charter here, what are the fees at Van Nuys?")])
    d = next(m for m in out if m.kind == "dispute_risk")
    r = tools.dispatch(d.tool_call)["result"]
    assert r["location_disputes"] == 0 and r["customer_disputes"] == 5 and r["customer_rank"] == 1
    assert "this customer has 5 (#1 of" in r["prompt_agent"]
    assert r["card"]["pattern"] == "Single key word, static data search"


def test_dispute_card_has_wording():
    _, out = run([("caller", "Northwind here, quote for KTEB.")])
    r = tools.dispatch(next(m for m in out if m.kind == "dispute_risk").tool_call)["result"]
    assert r["this_customer_here"] == 4 and r["top_reason"] == "After-hours callout fee"
    assert "Last issue here: After-hours callout fee" in r["prompt_agent"]
    assert r["quote_wording"].startswith("Fuel price excludes location fees.")


# --- Customer Experience: sentiment (model) and orchestration ---

class StubModel:
    """Stands in for Winnow: scores lines it was told about, regardless of wording."""
    name = "winnow"

    def __init__(self, scores):
        self.scores, self.calls = scores, []

    def score(self, speaker, latest, recent):
        self.calls.append((speaker, latest, list(recent)))
        return {"score": self.scores.get(latest, 0.0), "backend": self.name}


def test_sentiment_uses_the_model_not_keywords():
    # No lexicon word here, but the model hears frustration: that's the point of using a model.
    model = StubModel({"Is anyone actually going to fix this.": -0.8, "I have asked three times now.": -0.9})
    _, out = run([("caller", "Is anyone actually going to fix this."), ("caller", "I have asked three times now.")], scorer=model)
    alert = next(m for m in out if m.kind == "sentiment_shift")
    assert alert.tool_call["args"]["scored_by"] == "winnow"
    assert model.calls[1][2] == ["Is anyone actually going to fix this."]  # the model sees the speaker's recent lines


def test_sentiment_orchestration_alerts_agent_and_managers(settings, monkeypatch):
    sent = []
    monkeypatch.setattr("copilot.usecases._post", lambda url, body: sent.append((url, body)) or True)
    settings.MANAGER_WEBHOOKS = ["https://hooks.example.test/lead-a", "https://hooks.example.test/lead-b"]
    model = StubModel({"This is a joke.": -0.9, "Unbelievable.": -0.9})
    _, out = run([("caller", "This is a joke."), ("caller", "Unbelievable.")], scorer=model)
    r = tools.dispatch(next(m for m in out if m.kind == "sentiment_shift").tool_call)["result"]
    assert "caller's sentiment is turning negative" in r["agent_prompt"]
    assert r["managers_notified"] is True and [u for u, _ in sent] == settings.MANAGER_WEBHOOKS
    assert r["card"]["outcome"].startswith("Agent alerted on the call; designated managers")


# --- the Winnow client itself (server mocked) ---

class FakeResponse(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_winnow_request_and_score_mapping(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout):
        seen["url"], seen["body"] = req.full_url, json.loads(req.data)
        return FakeResponse(json.dumps({"answers": {"mood": {"type": "score", "probabilities": {"0": 0.0, "1": 0.1, "2": 0.2, "3": 0.7},
                                                              "expected": 2.6, "confidence": 0.5}}}).encode())

    monkeypatch.setattr(sentiment.urllib.request, "urlopen", fake_urlopen)
    w = WinnowSentiment("http://127.0.0.1:8091", api_key="k")
    r = w.score("caller", "That was unacceptable.", ["I'm calling about the invoice."])
    assert seen["url"] == "http://127.0.0.1:8091/v1/systemone"
    q = seen["body"]["questions"]["mood"]
    assert q["type"] == "score" and len(q["criteria"]) == 4
    assert seen["body"]["state"]["latest_line"] == "That was unacceptable."
    assert r["backend"] == "winnow" and r["score"] == pytest.approx(-0.8)


def test_winnow_expected_from_probabilities_only():
    assert expected_value({"probabilities": {"0": 0.5, "1": 0.5}})[0] == pytest.approx(0.5)


def test_winnow_down_falls_back_to_lexicon(monkeypatch):
    def boom(req, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr(sentiment.urllib.request, "urlopen", boom)
    w = WinnowSentiment("http://127.0.0.1:9")
    r = w.score("caller", "That was unacceptable.", [])
    assert r["score"] == -1.0 and r["backend"] == "lexicon (winnow unavailable)"


def test_scorer_selection(settings):
    sentiment.set_scorer(None)
    settings.SENTIMENT_BACKEND, settings.WINNOW_URL = "auto", ""
    assert isinstance(sentiment.get_scorer(), LexiconSentiment)
    sentiment.set_scorer(None)
    settings.WINNOW_URL = "http://winnow:8091"
    assert isinstance(sentiment.get_scorer(), WinnowSentiment)
    sentiment.set_scorer(None)
