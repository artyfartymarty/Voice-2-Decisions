"""Use cases from the sheet: GP quote lookup, avoiding disputes, customer-experience sentiment alerts."""
import json

import pytest

from copilot import reference, tools, usecases
from copilot.models import Call
from copilot.winnow import Winnower, sentiment, summarize

QUOTE_CALL = (
    "Hi, this is Dana with Northwind Aviation, account 204511. "
    "We're flying into Teterboro, K T E B, on Thursday. "
    "I need a fuel quote, what's your price per gallon? "
    "Honestly I'm frustrated, last time we were overcharged on the callout fee. "
    "That was unacceptable and really disappointing. "
    "Can you make sure the fees are clear this time?"
)


def run(text, **kw):
    w = Winnower(dispute_min_count=3, **kw)
    return w, w.feed(text, final=True)


def kinds(moments, only=("quote_request", "dispute_risk", "sentiment_shift")):
    return [m.kind for m in moments if m.kind in only]


# --- reference data ---

def test_reference_parses_messy_html_notes():
    ref = reference.get()
    kteb = ref.locations["KTEB"]
    assert kteb.fbo == "Meadowlark Jet Center"
    assert kteb.dispute_count == 7
    assert "After-hours callout fee: $175" in kteb.fees
    assert "<li>" not in kteb.public_notes and "&amp;" not in kteb.private_notes
    assert "call FBO direct for <2hr requests" in kteb.public_notes  # &lt; unescaped
    assert ref.locations["KDAL"].dispute_count == 3
    assert "De-ice fee: $1.10/gal Type I" in ref.locations["KDAL"].fees
    assert ref.locations["KVNY"].dispute_count == 0


# --- winnow raising use-case moments ---

def test_quote_call_raises_all_three_use_cases():
    w, moments = run(QUOTE_CALL)
    assert kinds(moments) == ["quote_request", "dispute_risk", "sentiment_shift"]
    quote = next(m for m in moments if m.kind == "quote_request")
    assert quote.tool_call["name"] == "quote_lookup"
    assert quote.tool_call["args"]["customer_number"] == "204511"
    assert quote.tool_call["args"]["icao"] == "KTEB"
    assert w.context["customer_name"] == "NORTHWIND AVIATION, LLC"


def test_no_quote_without_quote_intent():
    _, moments = run("This is Northwind Aviation. We land at KTEB on Thursday.")
    assert kinds(moments) == []


def test_quote_waits_for_customer():
    w = Winnower(dispute_min_count=3)
    first = w.feed("What's your price per gallon at KVNY?")
    assert kinds(first) == []  # KVNY has no disputes and we don't know who's calling yet
    later = w.feed("Sorry, this is Summit Ridge Flight Ops.")
    assert kinds(later) == ["quote_request"]
    assert later[-1].tool_call["args"] == {"text": later[-1].text, "customer_number": "129907", "icao": "KVNY"}


def test_dispute_guidance_only_for_high_dispute_locations():
    assert kinds(run("Can I get a quote for KVNY?")[1]) == []
    assert kinds(run("Can I get a quote for KTEB?")[1]) == ["dispute_risk"]
    assert kinds(run("Can I get a quote for EGGW?")[1]) == []  # 1 dispute, below the threshold of 3


def test_each_location_quoted_and_warned_once():
    _, moments = run("Northwind Aviation here. Quote for KTEB please. And the price at KTEB again?")
    assert kinds(moments) == ["quote_request", "dispute_risk"]


def test_sentiment_lexicon():
    assert sentiment("That was unacceptable.") == -1
    assert sentiment("Thanks, that's perfect.") == 1
    assert sentiment("We land at noon.") == 0


def test_one_bad_line_does_not_alert():
    assert kinds(run("Thanks for the help. I'm a bit frustrated with the delay. Anyway, sounds good.")[1]) == []


def test_sentiment_alert_and_rearm():
    w = Winnower()
    out = w.feed("This is ridiculous. I'm really upset.")
    assert kinds(out) == ["sentiment_shift"]
    assert kinds(w.feed("Honestly unacceptable.")) == []  # no repeat while still negative
    w.feed("Thank you, that's great. Perfect, I appreciate it. Wonderful.")
    assert w.context["alerted"][""] is False
    assert kinds(w.feed("Now this is terrible. Awful, just awful. I'm furious.")) == ["sentiment_shift"]


def test_sentiment_is_tracked_per_speaker():
    # The agent stays friendly; without per-speaker tracking that would hide the caller turning negative.
    w = Winnower()
    out = []
    for speaker, line in [("agent", "Thanks for calling, happy to help."), ("caller", "I'm frustrated with the last order."),
                          ("agent", "I appreciate you telling me."), ("caller", "Honestly it was unacceptable.")]:
        out += w.feed(line, final=True, speaker=speaker)
    alerts = [m for m in out if m.kind == "sentiment_shift"]
    assert [m.speaker for m in alerts] == ["caller"]
    assert alerts[0].tool_call["args"]["speaker"] == "caller"


def test_i_will_is_owned_by_the_speaker():
    out = Winnower().feed("I will send you the quote by end of day.", final=True, speaker="agent")
    action = next(m for m in out if m.kind == "action_item")
    assert (action.owner, action.due) == ("agent", "end of day")


def test_context_survives_partials_like_a_live_call():
    w1 = Winnower(dispute_min_count=3)
    w1.feed("This is Blue Heron Jet Charter.")
    w2 = Winnower(context=json.loads(json.dumps(w1.context)), dispute_min_count=3)  # stored as JSON in between
    assert kinds(w2.feed("How much per gallon at LFPB?")) == ["quote_request", "dispute_risk"]


# --- the tools ---

def test_quote_lookup_tool():
    r = tools.dispatch({"name": "quote_lookup", "args": {"text": "x", "customer_number": "204511", "icao": "KTEB"}})
    assert r["status"] == "done"
    res = r["result"]
    assert res["found"] is True
    assert res["addressable_volume_gal"] == 39500
    assert res["offer_range_gppg"] == [0.085, 0.3125]
    assert "KTEB" in res["dispatcher_alert"] and "39,500 gal addressable" in res["dispatcher_alert"]
    assert res["dispatcher_notified"] is False  # no webhook configured
    assert any("Minimum notice" in line for line in res["location_info"])


def test_quote_lookup_unknown_site():
    res = usecases.quote_lookup("x", "204511", "LFPB")
    assert res["found"] is False
    assert set(res["customer_locations"]) == {"KTEB", "KVNY", "EGGW"}


def test_quote_lookup_zero_addressable_volume_warns():
    res = usecases.quote_lookup("x", "377205", "KDAL")
    assert res["volume_check"] == "no addressable volume" and "Yield" in res["card"]["outcome"]


def test_dispute_guidance_tool():
    res = usecases.dispute_guidance("x", "KTEB")
    assert res["dispute_count"] == 7
    assert "After-hours callout fee: $175" in res["prompt_agent"]
    assert res["quote_wording"].startswith("Fuel price excludes location fees. Fees at Meadowlark Jet Center (KTEB)")


def test_alert_manager_posts_to_webhook(settings, monkeypatch):
    sent = {}

    class Resp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout):
        sent["url"], sent["body"] = req.full_url, json.loads(req.data)
        return Resp()

    settings.NOTIFY_WEBHOOK_URL = "https://hooks.example.test/abc"
    monkeypatch.setattr(usecases.urllib.request, "urlopen", fake_urlopen)
    res = usecases.alert_manager("That was unacceptable.", -1.0, ["a", "b"])
    assert res["managers_notified"] is True
    assert sent["url"] == "https://hooks.example.test/abc"
    assert sent["body"]["audience"] == "managers"


def test_summary_lists_use_cases():
    w, moments = run(QUOTE_CALL)
    dicts = []
    for m in moments:
        d = m.as_dict()
        d["tool_result"] = tools.dispatch(m.tool_call)["result"]
        dicts.append(d)
    s = summarize(dicts)
    assert s["quotes"][0]["offer_range_gppg"] == [0.085, 0.3125]
    assert s["dispute_warnings"][0]["dispute_count"] == 7
    assert s["sentiment_alerts"][0]["alerted"] is True


# --- through the API, as a live call ---

@pytest.mark.django_db
def test_live_quote_call_through_api(client):
    call = client.post("/api/calls/live/", json.dumps({"title": "Quote"}), content_type="application/json").json()
    url = f"/api/calls/{call['id']}/partial/"
    seen = []
    for piece in [QUOTE_CALL[:60], QUOTE_CALL[60:150], QUOTE_CALL[150:260], QUOTE_CALL[260:]]:
        r = client.post(url, json.dumps({"text": piece}), content_type="application/json").json()
        seen += [m["kind"] for m in r["new_moments"]]
    client.post(url, json.dumps({"text": "", "final": True}), content_type="application/json")
    assert {"quote_request", "dispute_risk", "sentiment_shift"} <= set(seen)

    s = client.get(f"/api/calls/{call['id']}/summary/").json()
    assert s["quotes"][0]["customer"] == "NORTHWIND AVIATION, LLC"
    assert Call.objects.get(pk=call["id"]).context["icao"] == "KTEB"
    page = client.get(f"/calls/{call['id']}/").content.decode()
    assert "0.085–0.3125 GPPG" in page and "Dispute risk:" in page and "Use cases" in page and "Managers alerted" in page
