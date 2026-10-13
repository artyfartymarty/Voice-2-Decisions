"""Contract tests: the live API exactly as its callers use it.

Callers, and what they send and read:
- five9-bridge (bridge.py, GATEWAY_MODE=django): POST /api/calls/live/ with its own JSON fields (ani may be null,
  plus a Bearer header) and reads "id"; then 8 kHz mono WAV chunks as multipart "audio" named chunkNNNNN.wav,
  "final" only on the last chunk, which can be empty when the legs end on a chunk boundary.
- the WAV test page: {"title"} -> "id"; 16 kHz chunks named chunk.wav; reads "status".
- scripts/stream_wav.py: reads "id", then "new_moments" (kind, text) and "speakers". Run here for real, over HTTP.
- scripts/smoke_test.sh: reads the summary keys in SUMMARY_KEYS.
If one of these tests fails, a caller breaks: change the caller in the same change, or keep the API compatible.
"""
import io
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
from django.core.files.uploadedfile import SimpleUploadedFile

from tests.conftest import tone

PARTIAL_KEYS = {"new_moments", "pending_text", "status", "speakers", "warnings"}
MOMENT_KEYS = {"kind", "text", "start", "end", "speaker", "tool_call"}
SUMMARY_KEYS = {"headline", "counts", "status", "quotes", "dispute_warnings", "sentiment_alerts", "speakers",
                "speaker_roles_confident", "warnings"}
BRIDGE_START = {"source": "five9", "call_id": "f9-123", "ani": None, "dnis": "8005551212", "agent": "mark",
                "agent_id": "a1", "domain_id": "d1", "campaign_id": "c1", "campaign": "Fuel", "skill_id": "s1",
                "direction": "inbound", "sample_rate": 8000, "encoding": "pcm_s16le", "frame_ms": 20,
                "title": "Five9 f9-123"}


def pcm_wav(samples: np.ndarray, rate: int) -> bytes:
    """A WAV the way bridge.py and the test page build one: mono, 16-bit, wave module / hand-written header."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


@pytest.mark.django_db
def test_bridge_django_mode(client):
    r = client.post("/api/calls/live/", json.dumps(BRIDGE_START), content_type="application/json",
                    HTTP_AUTHORIZATION="Bearer gateway-key")
    assert r.status_code == 201
    app_id = r.json()["id"]
    assert isinstance(app_id, int) and r.json()["status"] == "live"
    url = f"/api/calls/{app_id}/partial/"
    one_second = tone(1.0, sr=8000)
    for n in range(3):  # not final: the bridge leaves the field out entirely
        r = client.post(url, {"audio": SimpleUploadedFile(f"chunk{n:05d}.wav", pcm_wav(one_second, 8000), "audio/wav")})
        assert r.status_code == 200, r.content
        assert PARTIAL_KEYS <= set(r.json()) and r.json()["status"] == "live"
    empty_final = pcm_wav(np.zeros(0), 8000)
    r = client.post(url, {"audio": SimpleUploadedFile("chunk00003.wav", empty_final, "audio/wav"), "final": "true"})
    assert r.status_code == 200, r.content
    assert r.json()["status"] == "done"


@pytest.mark.django_db
def test_test_page(client):
    call = client.post("/api/calls/live/", json.dumps({"title": "Test: quote_call_mono.wav"}),
                       content_type="application/json").json()
    url = f"/api/calls/{call['id']}/partial/"
    r = client.post(url, {"audio": SimpleUploadedFile("chunk.wav", pcm_wav(tone(2.0), 16000), "audio/wav")})
    assert r.status_code == 200 and PARTIAL_KEYS <= set(r.json())
    r = client.post(url, {"audio": SimpleUploadedFile("chunk.wav", pcm_wav(np.zeros(0), 16000), "audio/wav"),
                          "final": "true"})
    assert r.status_code == 200 and r.json()["status"] == "done"


@pytest.mark.django_db
def test_text_partials_and_summary_shape(client):
    call = client.post("/api/calls/live/", json.dumps({"title": "t"}), content_type="application/json").json()
    url = f"/api/calls/{call['id']}/partial/"
    r = client.post(url, json.dumps({"text": "We decided to go with vendor B.", "speaker": "A", "final": True}),
                    content_type="application/json").json()
    assert PARTIAL_KEYS <= set(r) and MOMENT_KEYS <= set(r["new_moments"][0])
    assert SUMMARY_KEYS <= set(client.get(f"/api/calls/{call['id']}/summary/").json())


@pytest.mark.django_db
def test_partial_to_unknown_or_finished_call(client):
    assert client.post("/api/calls/999999/partial/", json.dumps({"text": "hi"}), content_type="application/json").status_code == 404
    bad = client.post("/api/calls/live/", json.dumps({}), content_type="application/json").json()
    r = client.post(f"/api/calls/{bad['id']}/partial/", {"audio": SimpleUploadedFile("x.wav", b"not audio")})
    assert r.status_code == 400 and "error" in r.json()


@pytest.mark.django_db(transaction=True)
def test_stream_wav_script_over_http(live_server, monkeypatch, capsys, settings):
    """scripts/stream_wav.py, unmodified, against a real server."""
    settings.ALLOWED_HOSTS = ["*"]
    sys.path.insert(0, str(Path(settings.BASE_DIR) / "scripts"))
    try:
        import stream_wav
    finally:
        sys.path.pop(0)
    monkeypatch.setattr(sys, "argv", ["stream_wav.py", str(Path(settings.BASE_DIR) / "tests/fixtures/sample_call.wav"),
                                      "--base", live_server.url, "--no-pace", "--chunk", "5", "--json"])
    stream_wav.main()
    report = json.loads(capsys.readouterr().out)
    assert report["chunks"] > 1 and report["latency_ms"]["p50"] >= 0
