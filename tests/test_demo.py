"""Demo mode (ASR_BACKEND=demo): each test call is transcribed from the script next to its WAV, chunk by chunk."""
import io
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from django.conf import settings as django_settings

from copilot import audio, demo, pipeline
from copilot.asr import set_backend
from copilot.diarize import set_diarizer
from copilot.models import Call

FIXTURES = Path(__file__).parent / "fixtures"
DEMOS = sorted(p.name.removesuffix(".script.json") for p in FIXTURES.glob("demo_*.script.json"))


@pytest.fixture
def demo_mode(settings):
    settings.TEST_CALLS_DIR = FIXTURES
    settings.DIARIZER = "demo"
    set_backend(demo.DemoASR())
    set_diarizer(demo.DemoDiarizer())
    yield
    set_backend(None)
    set_diarizer(None)


def chunks(name, seconds=2.0):
    """What the entry page posts: the WAV as 16 kHz mono PCM16 WAV chunks, slightly altered by browser resampling."""
    samples = audio.decode((FIXTURES / f"{name}.wav").read_bytes())
    samples = audio.resample(audio.resample(samples, 16000, 48000), 48000, 16000)
    step = int(seconds * 16000)
    for i in range(0, len(samples), step):
        buf = io.BytesIO()
        sf.write(buf, samples[i:i + step], 16000, format="WAV", subtype="PCM_16")
        yield buf.getvalue(), i + step >= len(samples)


def words(text):
    return [w.strip(".,?!").lower() for w in text.split()]


def test_every_demo_call_has_a_script_and_truth():
    assert len(DEMOS) >= 3
    for name in DEMOS:
        script = json.loads((FIXTURES / f"{name}.script.json").read_text())
        assert {line["speaker"] for line in script} == {"agent", "caller"}
        assert (FIXTURES / f"{name}.truth.json").exists()
        assert script[-1]["end"] <= sf.info(str(FIXTURES / f"{name}.wav")).duration


@pytest.mark.django_db
@pytest.mark.parametrize("name", DEMOS)
def test_streamed_demo_call_transcribes_its_script_with_roles(client, demo_mode, name):
    """Every word of the script comes back once, in order, and every line is labeled with the right role."""
    pk = client.post("/api/calls/live/", {"title": f"Test: {name}.wav"}, content_type="application/json").json()["id"]
    for data, final in chunks(name):
        form = {"audio": io.BytesIO(data)}
        form["audio"].name = "chunk.wav"
        if final:
            form["final"] = "true"
        assert client.post(f"/api/calls/{pk}/partial/", form).status_code == 200
    call = Call.objects.get(pk=pk)
    assert call.status == "done"
    script = json.loads((FIXTURES / f"{name}.script.json").read_text())
    said = [w for line in script for w in words(line["text"])]
    heard = [w for s in call.segments.all() for w in words(s.text)]
    assert heard == said
    roles = call.roles
    for seg in call.segments.all():
        mid = (seg.start + seg.end) / 2
        line = min(script, key=lambda l: 0 if l["start"] <= mid <= l["end"] else min(abs(mid - l["start"]), abs(mid - l["end"])))
        assert roles.get(seg.speaker) == line["speaker"], (seg.text, seg.speaker, roles)


@pytest.mark.django_db
def test_uploaded_demo_call_gets_the_use_cases(demo_mode):
    call = Call.objects.create(title="demo upload")
    pipeline.process_recording(call, [(FIXTURES / "demo_frustrated_caller.wav").read_bytes()])
    assert call.status == "done", call.error
    kinds = set(call.moments.values_list("kind", flat=True))
    assert {"quote_request", "dispute_risk", "sentiment_shift", "agent_prompt", "tail_number"} <= kinds
    assert call.context["tail"]["value"] == "N512BT"


@pytest.mark.django_db
def test_audio_that_matches_no_script_transcribes_to_nothing(demo_mode):
    from .conftest import tone
    assert demo.DemoASR().transcribe(tone(2.0, freq=313.0)).text == ""


def test_demo_backend_and_diarizer_are_selected_by_settings(settings):
    from copilot import asr, diarize
    settings.ASR_BACKEND, settings.DIARIZER = "demo", "demo"
    set_backend(None)
    set_diarizer(None)
    try:
        assert asr.get_backend().name == "demo" and diarize.get_diarizer().name == "demo"
    finally:
        set_backend(None)
        set_diarizer(None)
