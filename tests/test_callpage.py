"""The call page a call handler watches during a live call: panels, prompt cards, slot hint, alerts view."""
import json

import numpy as np
import pytest

from copilot import callview
from copilot.models import Call

from .test_usecases import QUOTE_CALL

pytestmark = pytest.mark.django_db


def live_call(client, **body):
    call = client.post("/api/calls/live/", json.dumps(body), content_type="application/json").json()
    return call["id"], f"/api/calls/{call['id']}/partial/"


def say(client, url, text, speaker="", final=False):
    return client.post(url, json.dumps({"text": text, "speaker": speaker, "final": final}), content_type="application/json").json()


def panels(client, pk, version=None):
    return client.get(f"/calls/{pk}/panels/" + (f"?v={version}" if version else ""))


def test_who_is_calling_before_and_after_customer_is_heard(client):
    pk, url = live_call(client)
    page = client.get(f"/calls/{pk}/").content.decode()
    assert "Customer not identified yet." in page

    say(client, url, "Hi, this is Dana with Northwind Aviation, account 204511. We're flying into Teterboro, K T E B.")
    who = callview.who_is_calling(Call.objects.get(pk=pk))
    assert who["customer_name"] == "NORTHWIND AVIATION, LLC" and who["customer_source"] == "heard on the call"
    assert who["icao"] == "KTEB" and who["fbo"] == "Meadowlark Jet Center" and who["on_file_here"] is True
    assert set(who["other_locations"]) == {"KVNY", "EGGW"}
    page = client.get(f"/calls/{pk}/").content.decode()
    assert "NORTHWIND AVIATION, LLC" in page and "Meadowlark Jet Center" in page


def test_prefilled_customer_shows_phone_system_source(client):
    pk, _ = live_call(client, customer_id="318842", ani="+12015550123")
    page = client.get(f"/calls/{pk}/").content.decode()
    assert "BLUE HERON JET CHARTER, INC" in page and "phone system" in page and "+12015550123" in page


def test_slot_hint_follows_the_tail_number(client):
    pk, url = live_call(client, customer_id="204511")
    assert callview.slot_hint(Call.objects.get(pk=pk)) is None

    say(client, url, "Tail number is bravo tango.", "caller")
    hint = callview.slot_hint(Call.objects.get(pk=pk))
    assert hint["state"] == "incomplete" and hint["value"] == "BT"
    assert "Tail number incomplete" in hint["message"] and "N512BT" in hint["message"]
    assert 'class="slot slot-incomplete"' in client.get(f"/calls/{pk}/").content.decode()

    say(client, url, "Sorry, what was the tail number again?", "agent")
    assert callview.slot_hint(Call.objects.get(pk=pk))["state"] == "waiting"

    say(client, url, "November five one two bravo tango.", "caller")
    hint = callview.slot_hint(Call.objects.get(pk=pk))
    assert hint == {"state": "confirmed", "label": "Tail number", "value": "N512BT", "message": "Tail N512BT confirmed."}


def test_prompt_cards_are_grouped_by_use_case(client):
    pk, url = live_call(client)
    groups = {g["title"]: g for g in callview.prompt_cards(Call.objects.get(pk=pk))}
    assert list(groups) == ["GP", "Avoiding Disputes", "Customer Experience", "Confirm on the call"]
    assert all(g["cards"] == [] for g in groups.values())
    assert "Nothing yet from: GP, Avoiding Disputes, Customer Experience, Confirm on the call." in client.get(f"/calls/{pk}/").content.decode()

    say(client, url, QUOTE_CALL, "caller")
    groups = {g["title"]: g for g in callview.prompt_cards(Call.objects.get(pk=pk))}
    gp = groups["GP"]["cards"][0]
    assert gp["headline"].startswith("Offer 0.0850 to 0.3125 GPPG")
    assert gp["category"] == "GP" and gp["who"] == "Customer" and gp["sent_to"] == "Dispatcher"
    assert "39,500 gal addressable" in gp["want_to_see"]
    dispute = groups["Avoiding Disputes"]["cards"][0]
    assert dispute["headline"].startswith("Dispute risk:") and "Fuel price excludes location fees" in dispute["say"]
    cx = groups["Customer Experience"]["cards"][0]
    assert cx["headline"].startswith("Heads up:") and cx["sent_to"] == "Call handler, managers / leads"


def test_transcript_has_speaker_labels(client):
    pk, url = live_call(client)
    say(client, url, "Good morning, fuel desk, how can I help?", "speaker 1")
    say(client, url, "Hi, this is Northwind Aviation, I need a fuel quote for Teterboro.", "speaker 2")
    turns = callview.transcript(Call.objects.get(pk=pk))
    assert [t["role"] for t in turns] == ["agent", "caller"]
    page = client.get(f"/calls/{pk}/").content.decode()
    assert 'class="turn r-agent"' in page and 'class="turn r-caller"' in page


def test_pending_sentence_shows_under_the_transcript(client):
    pk, url = live_call(client)
    say(client, url, "We're flying into")
    page = client.get(f"/calls/{pk}/").content.decode()
    assert "Waiting for the end of the sentence" in page and 'title="We&#x27;re flying into"' in page


def test_panels_endpoint_returns_html_and_skips_unchanged(client):
    pk, url = live_call(client)
    r = panels(client, pk)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "live" and set(data["panels"]) == {"who", "slot", "prompts", "transcript", "summary"}
    assert data["panels"]["prompts"].startswith('<div class="card" id="panel-prompts">')
    assert panels(client, pk, data["version"]).status_code == 204

    say(client, url, "This is Northwind Aviation, quote for KTEB please.", "caller")
    r = panels(client, pk, data["version"])
    assert r.status_code == 200 and r.json()["version"] != data["version"]
    assert "NORTHWIND AVIATION, LLC" in r.json()["panels"]["who"]


def test_panels_for_missing_call_is_404(client):
    assert panels(client, 999).status_code == 404


def test_summary_appears_when_the_call_ends(client):
    pk, url = live_call(client)
    say(client, url, QUOTE_CALL, "caller")
    assert "Call summary" not in panels(client, pk).json()["panels"]["summary"]
    say(client, url, "", final=True)
    data = panels(client, pk).json()
    assert data["status"] == "done"
    summary = data["panels"]["summary"]
    assert "Call summary" in summary and "Use cases" in summary and "Dispatcher" in summary


def test_live_page_has_the_refresh_script(client, settings):
    pk, url = live_call(client)
    page = client.get(f"/calls/{pk}/").content.decode()
    assert f"/calls/{pk}/panels/" in page and "refreshMs: 1500" in page and "new WebSocket" in page
    assert 'wsUrl: ""' in page  # no gateway configured: poll only

    settings.GATEWAY_WS_URL = "wss://gateway.example/v1/events"
    page = client.get(f"/calls/{pk}/").content.decode()
    assert "wss://gateway.example/v1/events" in page

    say(client, url, "", final=True)
    assert "v2dLive(" not in client.get(f"/calls/{pk}/").content.decode()


def test_manager_alerts_view(client):
    done_pk, done_url = live_call(client, title="Finished call")
    say(client, done_url, QUOTE_CALL, "caller", final=True)
    live_pk, live_url = live_call(client, title="Angry caller")
    say(client, live_url, "Honestly I'm frustrated, this is unacceptable. That was really disappointing.", "caller")

    alerts = callview.manager_alerts()
    assert {(a["call_id"], a["kind"]) for a in alerts} >= {(done_pk, "quote_request"), (done_pk, "sentiment_shift"),
                                                           (live_pk, "sentiment_shift")}
    page = client.get("/alerts/").content.decode()
    assert "Manager alerts" in page and "Angry caller" in page and "Finished call" in page
    assert f'href="/calls/{live_pk}/"' in page and "Sentiment turning negative" in page
    assert "is asking for a quote at KTEB" in page  # the dispatcher alert text

    live_only = client.get("/alerts/?live=1").content.decode()
    assert "Angry caller" in live_only and "Finished call" not in live_only

    fragment = client.get("/alerts/?partial=1").content.decode()
    assert fragment.lstrip().startswith('<div id="panel-alerts">') and "<html" not in fragment


def test_manager_alerts_empty(client):
    assert "No alerts." in client.get("/alerts/").content.decode()


# --- entry page: test with a recording ---

def test_entry_page_lists_the_test_calls(client):
    page = client.get("/").content.decode()
    assert "Drop a WAV file" in page and "Test calls" in page
    assert 'data-url="/test-calls/quote_call_mono.wav"' in page and "8 labeled turns" in page
    assert "/api/calls/live/" in page and "/partial/" in page  # the browser streams the file as a live call


def test_test_calls_list_reads_the_folder(settings, tmp_path):
    from .conftest import tone, wav_bytes
    (tmp_path / "b.wav").write_bytes(wav_bytes(tone(2.0)))
    (tmp_path / "a.wav").write_bytes(wav_bytes(tone(1.0)))
    (tmp_path / "a.truth.json").write_text('[{"start": 0, "end": 1, "role": "agent"}]')
    (tmp_path / "broken.wav").write_bytes(b"not audio")
    (tmp_path / "notes.txt").write_text("ignored")
    settings.TEST_CALLS_DIR = tmp_path
    calls = callview.test_calls()
    assert [(c["name"], c["duration"], c["labeled_turns"]) for c in calls] == [("a.wav", 1.0, 1), ("b.wav", 2.0, 0)]
    settings.TEST_CALLS_DIR = tmp_path / "missing"
    assert callview.test_calls() == []


def test_test_call_file_is_served(client):
    r = client.get("/test-calls/quote_call_mono.wav")
    assert r.status_code == 200 and r["Content-Type"] == "audio/wav"
    assert b"".join(r.streaming_content)[:4] == b"RIFF"


def test_test_call_file_only_serves_listed_files(client):
    assert client.get("/test-calls/nope.wav").status_code == 404
    assert client.get("/test-calls/quote_call_mono.truth.json").status_code == 404
    assert client.get("/test-calls/..%2Fconftest.py").status_code == 404


def test_streamed_chunks_build_a_live_call(client, fake_asr):
    """What the entry page does: start a live call, post PCM16 WAV chunks in order, the last one with final."""
    from django.core.files.uploadedfile import SimpleUploadedFile
    from .conftest import tone, wav_bytes
    pk, url = live_call(client, title="Test: quote_call_mono.wav")
    lines = ["Northwind Aviation here, account 204511.", "I need a fuel quote at KTEB please.", ""]
    for i, text in enumerate(lines):
        fake_asr.text = text
        # Each chunk ends in a pause, as speech does: the live path only transcribes up to the last pause.
        chunk = np.concatenate([tone(1.5), np.zeros(8000, dtype=np.float32)])
        data = {"audio": SimpleUploadedFile("chunk.wav", wav_bytes(chunk), "audio/wav")}
        if i == len(lines) - 1:
            data["final"] = "true"
        assert client.post(url, data).status_code == 200
    call = Call.objects.get(pk=pk)
    assert call.status == "done" and call.duration == pytest.approx(6.0, abs=0.01)
    assert call.moments.filter(kind="quote_request").exists()


def test_pages_warn_when_the_speech_model_is_fake(client):
    pk, _ = live_call(client)
    for url in ("/", f"/calls/{pk}/"):
        assert "The speech model is off (ASR_BACKEND=fake)" in client.get(url).content.decode()


# --- fixes from the scripted demo run ---

def test_consecutive_lines_from_one_speaker_are_one_turn(client):
    pk, url = live_call(client)
    for spk, text in [("speaker 2", "Yeah, this"), ("speaker 2", "is Dana at Northwind"), ("speaker 2", "Aviation, account 204511."),
                      ("speaker 1", "Thanks Dana, how can I help?"), ("speaker 2", "I need a fuel quote.")]:
        say(client, url, text, spk)
    turns = callview.transcript(Call.objects.get(pk=pk))
    assert [t["text"] for t in turns] == ["Yeah, this is Dana at Northwind Aviation, account 204511.",
                                          "Thanks Dana, how can I help?", "I need a fuel quote."]
    assert turns[0]["start"] == 0.0 and turns[0]["id"] < turns[1]["id"]


def test_first_tail_ask_does_not_say_repeat():
    call = Call(context={"tail": {"value": None, "partial": "", "pending": True}})
    assert callview.slot_hint(call)["message"] == "Waiting for the caller to give the tail number."
    call.context["tail"]["partial"] = "BT"
    assert "repeat" in callview.slot_hint(call)["message"]


def test_tail_prompt_is_resolved_once_the_tail_is_confirmed(client):
    pk, url = live_call(client, customer_id="204511")
    say(client, url, "Tail number is bravo tango.", "caller")
    confirm = next(g for g in callview.prompt_cards(Call.objects.get(pk=pk)) if g["title"] == "Confirm on the call")
    assert [c["resolved"] for c in confirm["cards"]] == [False]
    say(client, url, "Sorry, what was the tail number again?", "agent")
    say(client, url, "November five one two bravo tango.", "caller")
    groups = callview.prompt_cards(Call.objects.get(pk=pk))
    assert groups[0]["title"] == "Confirm on the call"  # the group with the newest card is on top
    assert [(c["kind"], c["resolved"]) for c in groups[0]["cards"]] == [("tail_number", False), ("agent_prompt", True)]
    assert 'class="pcard k-agent_prompt resolved"' in client.get(f"/calls/{pk}/").content.decode()


def test_summary_headline_says_what_the_call_was_about(client):
    pk, url = live_call(client)
    say(client, url, QUOTE_CALL, "caller", final=True)
    line = callview.headline(Call.objects.get(pk=pk))
    assert line == "NORTHWIND AVIATION, LLC; quote wanted at KTEB; dispute risk: fees restated; caller upset, managers alerted"
    assert f"<b>{line}</b>" in panels(client, pk).json()["panels"]["summary"]


def test_headline_falls_back_without_use_cases(client, wav):
    from copilot import pipeline
    call = Call.objects.create()
    pipeline.process_recording(call, [wav])
    assert callview.headline(call) == ""
    assert "<b>We decided to go with the annual plan.</b>" in client.get(f"/calls/{call.pk}/").content.decode()


def test_header_shows_the_speech_backend(client, settings):
    settings.ASR_BACKEND = "demo"
    assert 'class="mode mode-demo"' in client.get("/alerts/").content.decode()


def test_test_calls_put_whole_calls_first(settings, tmp_path):
    from .conftest import tone, wav_bytes
    for name in ("a_agent.wav", "b.wav", "c.wav"):
        (tmp_path / name).write_bytes(wav_bytes(tone(1.0)))
    (tmp_path / "c.script.json").write_text("[]")
    settings.TEST_CALLS_DIR = tmp_path
    calls = callview.test_calls()
    assert [(c["name"], c["scripted"], c["single_leg"]) for c in calls] == [
        ("c.wav", True, False), ("b.wav", False, False), ("a_agent.wav", False, True)]


def test_alerts_have_a_time(client):
    pk, url = live_call(client, title="Angry caller")
    say(client, url, "Honestly I'm frustrated, this is unacceptable. That was really disappointing.", "caller")
    alert = callview.manager_alerts()[0]
    assert alert["alerted_at"] >= Call.objects.get(pk=pk).created_at
    assert "<time datetime=" in client.get("/alerts/").content.decode()
