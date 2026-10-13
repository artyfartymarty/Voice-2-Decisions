"""The speech-service backend (NeMo-Speech.cpp) with the HTTP server mocked."""
import io
import json

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile

from copilot.asr import NemoSpeechASR, set_backend
from copilot.diarize import FakeDiarizer, set_diarizer
from tests.conftest import turns_audio, wav_bytes

WORDS = [  # what verbose_json + diarization returns: words with times and 1-based speakers
    *[{"word": w, "start": 0.1 + 0.3 * i, "end": 0.35 + 0.3 * i, "speaker": 1} for i, w in enumerate("Thanks for calling. How can I help?".split())],
    *[{"word": w, "start": 3.0 + 0.3 * i, "end": 3.25 + 0.3 * i, "speaker": 2}
      for i, w in enumerate("Northwind Aviation here. I need a quote at KTEB, what's your price per gallon?".split())],
    *[{"word": w, "start": 8.0 + 0.3 * i, "end": 8.25 + 0.3 * i, "speaker": 1} for i, w in enumerate("Let me pull that up.".split())],
]


class FakeHTTP(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def server(monkeypatch):
    calls = []

    def urlopen(req, timeout):
        body = req.data.decode("latin-1")
        calls.append({"url": req.full_url, "diarization": 'name="diarization"\r\n\r\ntrue' in body,
                      "boost": "speech_contexts" in body, "auth": req.headers.get("Authorization")})
        words = WORDS if calls[-1]["diarization"] else [{**w, "speaker": None} for w in WORDS]
        return FakeHTTP(json.dumps({"text": " ".join(w["word"] for w in WORDS), "words": words}).encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    return calls


def test_segments_split_at_pauses(server):
    t = NemoSpeechASR("http://asr:8080").transcribe(turns_audio(1))
    assert [s.text for s in t.segments] == ["Thanks for calling. How can I help?",
                                            "Northwind Aviation here. I need a quote at KTEB, what's your price per gallon?",
                                            "Let me pull that up."]
    assert server[0]["url"] == "http://asr:8080/v1/audio/transcriptions" and server[0]["boost"]


def test_speakers_in_one_request(server):
    out = NemoSpeechASR("http://asr:8080", api_key="k").transcribe_with_speakers(turns_audio(1))
    assert [spk for _, spk in out] == ["speaker 1", "speaker 2", "speaker 1"]
    assert server[0]["diarization"] and server[0]["auth"] == "Bearer k"


@pytest.mark.django_db
def test_mono_recording_through_speech_service(client, server, settings):
    settings.DIARIZER = "nemotron"
    set_diarizer(FakeDiarizer())  # present but unused: the speech service supplies the speakers
    set_backend(NemoSpeechASR("http://asr:8080"))
    d = client.post("/api/calls/", {"audio": SimpleUploadedFile("m.wav", wav_bytes(turns_audio(3)))}).json()
    assert d["speakers"] == {"speaker 1": "agent", "speaker 2": "caller"}
    assert len(server) == 1 and server[0]["diarization"]  # one request did ASR + speakers
    assert {m["kind"] for m in d["moments"]} >= {"quote_request", "dispute_risk"}
