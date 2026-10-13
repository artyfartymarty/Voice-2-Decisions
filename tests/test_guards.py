"""Guardrails: the server refuses a fake or misspelled backend, doctor fails loudly on one, and transcripts that
can't be right (the same text over and over, or nothing from speech) are flagged on the call."""
import io
import json
from contextlib import redirect_stdout

import pytest
from django.core.exceptions import ImproperlyConfigured
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management import call_command

from copilot import golden, guards
from copilot.asr import FakeASR, set_backend
from tests.conftest import ScriptedASR, turns_audio, wav_bytes


# --- refusing to start ---

def test_fake_backend_refused_unless_allowed(settings):
    settings.ASR_BACKEND, settings.ALLOW_FAKE_BACKEND = "fake", False
    assert any("ASR_BACKEND=fake" in p for p in guards.config_problems())
    with pytest.raises(ImproperlyConfigured, match="ALLOW_FAKE_BACKEND=1"):
        guards.enforce()
    assert [e.id for e in guards.check_settings()] == ["v2d.E001"]
    settings.ALLOW_FAKE_BACKEND = True
    assert guards.config_problems() == []
    guards.enforce()  # allowed on purpose: starts, with a warning in the log


def test_fake_diarizer_refused_too(settings):
    settings.ASR_BACKEND, settings.DIARIZER, settings.ALLOW_FAKE_BACKEND = "nemo_speech", "fake", False
    assert any("DIARIZER=fake" in p for p in guards.config_problems())


def test_misspelled_backend_refused_at_start_not_first_request(settings):
    settings.ASR_BACKEND = "nemo_spech"
    assert guards.config_problems() == ["ASR_BACKEND='nemo_spech' is not one of nemo_speech, nemotron, demo, fake"]


def test_real_config_starts(settings):
    settings.ASR_BACKEND, settings.DIARIZER, settings.ALLOW_FAKE_BACKEND = "nemo_speech", "nemotron", False
    assert guards.config_problems() == []


def test_healthz_says_when_the_model_is_fake(client):
    d = client.get("/healthz").json()
    assert d["fake"] is True and "speech model is the fake one" in d["warnings"][0]
    set_backend(type("Real", (FakeASR,), {"name": "nemo_speech"})())
    d = client.get("/healthz").json()
    assert d["fake"] is False and d["warnings"] == []


# --- transcript sanity ---

def live(client):
    call = client.post("/api/calls/live/", json.dumps({}), content_type="application/json").json()
    return f"/api/calls/{call['id']}/partial/", call["id"]


@pytest.mark.django_db
def test_same_text_every_chunk_is_flagged(client):
    """The mistake that started this: a fake or stuck model gives the same paragraph for every chunk."""
    url, pk = live(client)
    one = wav_bytes(turns_audio(1))
    codes = []
    for i in range(4):
        r = client.post(url, {"audio": SimpleUploadedFile(f"c{i}.wav", one), "final": "true" if i == 3 else ""}).json()
        codes = [w["code"] for w in r["warnings"]]
    assert "repeated_text" in codes and "fake_backend" in codes
    summary = client.get(f"/api/calls/{pk}/summary/").json()
    assert {w["code"] for w in summary["warnings"]} >= {"repeated_text", "fake_backend"}
    assert client.get(f"/api/calls/{pk}/").json()["warnings"] == summary["warnings"]


def test_short_replies_may_repeat():
    from copilot.models import Call
    call = Call(context={})
    check = guards.TranscriptCheck(call)
    for _ in range(5):
        check.heard("Okay, yeah.")
    codes = lambda: [w["code"] for w in call.context.get("warnings", []) if w["code"] != "fake_backend"]
    assert codes() == []
    for _ in range(3):
        check.heard("We decided to go with the annual plan.")
    assert codes() == ["repeated_text"]


@pytest.mark.django_db
def test_speech_with_no_text_is_flagged(client):
    set_backend(ScriptedASR([]))  # every chunk has speech, the "model" hears nothing
    url, _ = live(client)
    one = wav_bytes(turns_audio(1))
    for i in range(3):
        r = client.post(url, {"audio": SimpleUploadedFile(f"c{i}.wav", one)}).json()
    assert [w["code"] for w in r["warnings"]] == ["no_text"]


@pytest.mark.django_db
def test_silence_with_no_text_is_fine(client):
    import numpy as np
    set_backend(ScriptedASR([]))
    url, _ = live(client)
    quiet = wav_bytes(np.zeros(32000, dtype=np.float32))
    for i in range(4):
        r = client.post(url, {"audio": SimpleUploadedFile(f"c{i}.wav", quiet)}).json()
    assert r["warnings"] == []


@pytest.mark.django_db
def test_normal_call_has_no_warnings(client, scripted):
    scripted(["Thanks for calling, this is Mark.", "Hi, I need a fuel quote.", "Let me pull that up."])
    url, pk = live(client)
    one = wav_bytes(turns_audio(1))
    for i in range(3):
        r = client.post(url, {"audio": SimpleUploadedFile(f"c{i}.wav", one), "final": "true" if i == 2 else ""}).json()
    assert [w["code"] for w in r["warnings"]] == ["fake_backend"]  # ScriptedASR is a fake; nothing else is wrong


@pytest.mark.django_db
def test_recording_with_speech_but_no_text_is_flagged(client):
    set_backend(ScriptedASR([]))
    d = client.post("/api/calls/", {"audio": SimpleUploadedFile("m.wav", wav_bytes(turns_audio(2)))}).json()
    assert [w["code"] for w in d["warnings"]] == ["no_text"]


# --- doctor ---

def doctor(*args):
    out = io.StringIO()
    with redirect_stdout(out):
        try:
            call_command("doctor", "--json", *args, stdout=out)
            code = 0
        except SystemExit as exc:
            code = exc.code
    return code, {c["name"]: c for c in json.loads(out.getvalue())["checks"]}


@pytest.mark.django_db
def test_doctor_fails_on_the_fake_model(settings):
    settings.ASR_BACKEND, settings.DIARIZER, settings.ALLOW_FAKE_BACKEND = "fake", "off", True
    code, checks = doctor("--golden")
    assert code == 1
    assert checks["speech probe"]["status"] == "FAIL" and "fake model" in checks["speech probe"]["detail"]
    assert checks["golden quote_call"]["status"] == "FAIL"
    assert checks["config"]["status"] == "WARN"


@pytest.mark.django_db
def test_doctor_fails_on_same_text_for_different_turns(settings):
    settings.ASR_BACKEND, settings.DIARIZER, settings.SECRET_KEY = "nemotron", "off", "a-real-secret"
    set_backend(type("Stuck", (FakeASR,), {"name": "nemotron"})(text="hello there"))
    _, checks = doctor()
    assert checks["speech probe"]["status"] == "FAIL" and "same text" in checks["speech probe"]["detail"]


@pytest.mark.django_db
def test_doctor_passes_a_working_model(settings):
    settings.ASR_BACKEND, settings.DIARIZER, settings.SECRET_KEY, settings.ASR_DEVICE = "nemotron", "off", "s3cret", "cpu"
    set_backend(type("Real", (ScriptedASR,), {"name": "nemotron"})(["Thanks for calling, this is Mark.", "Hi, it's Dana."]))
    code, checks = doctor()
    assert checks["speech probe"]["status"] == "PASS", checks["speech probe"]
    assert code == 0, [c for c in checks.values() if c["status"] == "FAIL"]


@pytest.mark.django_db
def test_doctor_reports_bad_config_instead_of_crashing(settings):
    settings.ASR_BACKEND = "nemo_spech"
    code, checks = doctor("--no-probe")
    assert code == 1 and checks["config"]["status"] == "FAIL"


@pytest.mark.django_db
def test_doctor_speech_service_down(settings):
    settings.ASR_BACKEND, settings.NEMO_SPEECH_URL, settings.DIARIZER = "nemo_speech", "http://127.0.0.1:9", "off"
    _, checks = doctor("--no-probe")
    assert checks["speech service"]["status"] == "FAIL" and "unreachable" in checks["speech service"]["detail"]


@pytest.mark.django_db
def test_golden_wav_through_live_path_runs(settings):
    """The --golden mechanics (chunked WAV through process_partial, rolled back) on the fake model: it must fail."""
    case = next(c for c in golden.load() if "wav" in c)
    result = golden.play_wav(case, chunk_s=4.0)
    assert len(result["transcript"]) >= 3
    assert any("same text" in f for f in golden.compare(case, result))
