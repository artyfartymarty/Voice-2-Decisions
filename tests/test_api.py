import json

import pytest
from django.core.management import call_command

from copilot import pipeline, tools
from copilot.models import Call, Moment

pytestmark = pytest.mark.django_db


def test_healthz(client):
    data = client.get("/healthz").json()
    assert data["ok"] is True
    assert data["asr"]["backend"] == "fake"
    assert "log_decision" in data["asr"]["tools"]


def test_upload_recording(client, wav):
    from django.core.files.uploadedfile import SimpleUploadedFile
    r = client.post("/api/calls/", {"title": "Acme", "audio": SimpleUploadedFile("a.wav", wav, "audio/wav")})
    assert r.status_code == 201, r.content
    data = r.json()
    assert data["status"] == "done"
    assert "annual plan" in data["transcript"]
    assert {m["kind"] for m in data["moments"]} == {"decision", "action_item", "risk", "question", "key_moment"}
    assert all(m["tool_status"] == "done" for m in data["moments"])

    s = client.get(f"/api/calls/{data['id']}/summary/").json()
    assert s["headline"] == "We decided to go with the annual plan."
    assert s["action_items"] == [{"text": "Sarah will send the contract by Friday.", "owner": "Sarah", "due": "Friday"}]


def test_two_legs_are_mixed(client, wav):
    from django.core.files.uploadedfile import SimpleUploadedFile
    r = client.post("/api/calls/", {"audio": SimpleUploadedFile("a.wav", wav), "audio_b": SimpleUploadedFile("b.wav", wav)})
    assert r.status_code == 201
    data = r.json()
    assert data["duration"] == pytest.approx(1.0, abs=0.01)
    assert [s["speaker"] for s in data["segments"]] == ["agent", "caller"]


def test_two_legs_custom_labels_and_mix_mode(client, wav, settings):
    from django.core.files.uploadedfile import SimpleUploadedFile
    r = client.post("/api/calls/", {"audio": SimpleUploadedFile("a.wav", wav), "audio_b": SimpleUploadedFile("b.wav", wav),
                                    "speaker_a": "dispatcher", "speaker_b": "pilot"}).json()
    assert [s["speaker"] for s in r["segments"]] == ["dispatcher", "pilot"]
    settings.LEG_MODE = "mix"
    r = client.post("/api/calls/", {"audio": SimpleUploadedFile("a.wav", wav), "audio_b": SimpleUploadedFile("b.wav", wav)}).json()
    assert [s["speaker"] for s in r["segments"]] == [""]  # one mixed leg, no labels


def test_upload_without_audio_is_400(client):
    assert client.post("/api/calls/", {}).status_code == 400


def test_bad_audio_marks_call_error(client):
    from django.core.files.uploadedfile import SimpleUploadedFile
    r = client.post("/api/calls/", {"audio": SimpleUploadedFile("x.wav", b"not audio at all" * 50)})
    assert r.status_code == 422
    assert r.json()["status"] == "error"


def test_live_call_with_text_partials(client):
    call = client.post("/api/calls/live/", json.dumps({"title": "Live"}), content_type="application/json").json()
    url = f"/api/calls/{call['id']}/partial/"

    r = client.post(url, json.dumps({"text": "We decided to go"}), content_type="application/json").json()
    assert r["new_moments"] == [] and r["pending_text"] == "We decided to go"

    r = client.post(url, json.dumps({"text": "with vendor B. Can you"}), content_type="application/json").json()
    assert [m["kind"] for m in r["new_moments"]] == ["decision"]

    r = client.post(url, json.dumps({"text": "send the quote", "final": True}), content_type="application/json").json()
    assert r["status"] == "done"
    assert [m["kind"] for m in r["new_moments"]] == ["action_item"]
    assert Call.objects.get(pk=call["id"]).moments.count() == 2


def test_live_call_with_audio_chunk(client, wav, fake_asr):
    from django.core.files.uploadedfile import SimpleUploadedFile
    fake_asr.text = "Let's go with the annual plan."
    call = client.post("/api/calls/live/", json.dumps({}), content_type="application/json").json()
    r = client.post(f"/api/calls/{call['id']}/partial/", {"audio": SimpleUploadedFile("c.wav", wav)})
    assert [m["kind"] for m in r.json()["new_moments"]] == ["decision"]
    assert Call.objects.get(pk=call["id"]).duration == pytest.approx(1.0, abs=0.01)


def test_pending_tool_calls_replay_once_tool_exists(client, wav):
    saved = tools.REGISTRY.pop("flag_risk")
    try:
        call = Call.objects.create()
        pipeline.process_recording(call, [wav])
        risk = call.moments.get(kind="risk")
        assert risk.tool_status == "pending"
    finally:
        tools.REGISTRY["flag_risk"] = saved
    assert pipeline.replay_pending(call) == 1
    assert Moment.objects.get(pk=risk.pk).tool_status == "done"


def test_pages_render(client, wav):
    call = Call.objects.create(title="Page test")
    pipeline.process_recording(call, [wav])
    assert b"Upload a call" in client.get("/").content
    page = client.get(f"/calls/{call.pk}/").content.decode()
    assert "We decided to go with the annual plan." in page
    assert "create_action_item" in page


def test_form_upload_redirects(client, wav):
    from django.core.files.uploadedfile import SimpleUploadedFile
    r = client.post("/", {"title": "Form", "audio": SimpleUploadedFile("a.wav", wav)})
    assert r.status_code == 302


def test_process_call_command(tmp_path, wav, capsys):
    f = tmp_path / "call.wav"
    f.write_bytes(wav)
    call_command("process_call", str(f))
    out = json.loads(capsys.readouterr().out)
    assert out["counts"]["decision"] == 1
