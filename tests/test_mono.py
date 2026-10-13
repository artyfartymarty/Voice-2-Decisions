"""Mono calls: diarization -> per-turn transcription -> agent/caller roles -> use cases."""
import json

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile

from copilot import extract
from copilot.diarize import Turn, clean_turns
from copilot.pipeline import _split_chunk_by_turns
from copilot.roles import infer_roles
from tests.conftest import turns_audio, wav_bytes

AGENT_1 = "Thanks for calling, this is Mark. How can I help?"
CALLER_1 = "Hi, this is Dana with Northwind Aviation. I need a fuel quote at KTEB, what's your price per gallon?"
AGENT_2 = "Let me pull that up for you."
CALLER_2 = "Honestly I'm frustrated, last time we were overcharged. That was unacceptable."
AGENT_3 = "I understand. I will send you the quote with every fee listed by end of day."
SCRIPT = [AGENT_1, CALLER_1, AGENT_2, CALLER_2, AGENT_3]


# --- roles ---

def test_roles_from_content():
    info = infer_roles([("speaker 1", AGENT_1), ("speaker 2", CALLER_1), ("speaker 1", AGENT_2)], extract.get())
    assert info["roles"] == {"speaker 1": "agent", "speaker 2": "caller"}
    assert info["confident"]


def test_roles_when_caller_speaks_first():
    # e.g. the recording starts after the greeting: content beats the first-speaker prior
    info = infer_roles([("speaker 1", CALLER_1), ("speaker 2", "Sure, let me pull that up for you. Our price is good.")])
    assert info["roles"] == {"speaker 1": "caller", "speaker 2": "agent"}


def test_roles_without_evidence_fall_back_to_first_speaker():
    info = infer_roles([("speaker 1", "Hello."), ("speaker 2", "Hi there.")])
    assert info["roles"] == {"speaker 1": "agent", "speaker 2": "caller"}
    assert not info["confident"]


def test_roles_third_speaker_keeps_its_name():
    info = infer_roles([("speaker 1", AGENT_1), ("speaker 2", CALLER_1), ("speaker 3", "Hold on, let me grab the tail number.")])
    assert info["roles"]["speaker 3"] == "speaker 3"


# --- turn handling ---

def test_clean_turns_trims_overlap_and_merges_short_pauses():
    turns = clean_turns([Turn(0, 2, "speaker 1"), Turn(2.3, 4, "speaker 1"), Turn(3.5, 6, "speaker 2"), Turn(5.9, 6.0, "speaker 1")])
    assert [(t.start, t.end, t.speaker) for t in turns] == [(0, 4, "speaker 1"), (4, 6, "speaker 2")]


def test_split_live_chunk_by_turns():
    turns = [Turn(0, 3, "speaker 1"), Turn(4, 7, "speaker 2")]
    # chunk 2..6: tail of speaker 1, a pause (goes to speaker 1), speaker 2 until the chunk ends
    assert _split_chunk_by_turns(2, 6, turns, "") == [(2, 4, "speaker 1"), (4, 6, "speaker 2")]
    # nothing diarized yet in this chunk: whoever spoke last
    assert _split_chunk_by_turns(8, 9, turns, "speaker 2") == [(8, 9, "speaker 2")]


def test_role_accuracy_scoring():
    from copilot.evaluation import role_accuracy
    truth = [{"start": 0, "end": 4, "role": "agent"}, {"start": 5, "end": 9, "role": "caller"}]
    segs = [{"start": 0, "end": 4, "role": "agent"}, {"start": 5, "end": 8, "role": "caller"}, {"start": 8, "end": 9, "role": "agent"}]
    s = role_accuracy(segs, truth)
    assert s["time_accuracy"] == pytest.approx(7 / 8)
    assert s["turns_correct"] == 2 and s["speech_covered"] == 1.0


# --- end to end with the fake diarizer ---

@pytest.mark.django_db
def test_mono_recording_gets_roles_and_use_cases(client, scripted):
    scripted(SCRIPT)
    r = client.post("/api/calls/", {"title": "mono", "audio": SimpleUploadedFile("m.wav", wav_bytes(turns_audio(5)))})
    assert r.status_code == 201, r.content
    d = r.json()
    assert d["speakers"] == {"speaker 1": "agent", "speaker 2": "caller"}
    assert [s["role"] for s in d["segments"]] == ["agent", "caller", "agent", "caller", "agent"]
    kinds = {m["kind"] for m in d["moments"]}
    assert {"quote_request", "dispute_risk", "sentiment_shift", "action_item"} <= kinds

    s = client.get(f"/api/calls/{d['id']}/summary/").json()
    assert s["sentiment_alerts"][0]["speaker"] == "caller"
    assert any(a["owner"] == "agent" and a["due"] == "end of day" for a in s["action_items"])
    page = client.get(f"/calls/{d['id']}/").content.decode()
    assert "caller</b> (speaker 2)" in page


@pytest.mark.django_db
def test_mono_without_diarizer_still_works(client, wav):
    r = client.post("/api/calls/", {"audio": SimpleUploadedFile("m.wav", wav)}).json()
    assert r["status"] == "done" and r["speakers"] == {}
    assert all(s["speaker"] == "" for s in r["segments"])


@pytest.mark.django_db
def test_live_mono_call_keeps_speakers_across_chunks(client, scripted):
    scripted([AGENT_1, CALLER_1, CALLER_2])
    call = client.post("/api/calls/live/", json.dumps({}), content_type="application/json").json()
    url = f"/api/calls/{call['id']}/partial/"
    one = wav_bytes(turns_audio(1))  # 1 s pause, 1 s speech, 1 s pause
    kinds = []
    for i in range(3):
        r = client.post(url, {"audio": SimpleUploadedFile(f"c{i}.wav", one), "final": "true" if i == 2 else ""}).json()
        kinds += [m["kind"] for m in r["new_moments"]]
    assert r["status"] == "done"
    d = client.get(f"/api/calls/{call['id']}/").json()
    assert [s["speaker"] for s in d["segments"]] == ["speaker 1", "speaker 2", "speaker 1"]  # fake diarizer alternates
    assert d["speakers"]["speaker 1"] in ("agent", "caller")
    assert "quote_request" in kinds


@pytest.mark.django_db
def test_text_partials_with_speaker_labels(client):
    call = client.post("/api/calls/live/", json.dumps({}), content_type="application/json").json()
    url = f"/api/calls/{call['id']}/partial/"
    for spk, line in [("A", AGENT_1), ("B", "I'm frustrated with the last order."), ("A", "I appreciate that."),
                      ("B", "It was unacceptable, honestly.")]:
        r = client.post(url, json.dumps({"text": line, "speaker": spk}), content_type="application/json").json()
    r = client.post(url, json.dumps({"final": True}), content_type="application/json").json()
    d = client.get(f"/api/calls/{call['id']}/summary/").json()
    assert d["speakers"] == {"A": "agent", "B": "caller"}
    assert [a["speaker"] for a in d["sentiment_alerts"]] == ["caller"]
