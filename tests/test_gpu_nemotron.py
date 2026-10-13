"""Real-model tests. Run with RUN_GPU_TESTS=1 (the model downloads from Hugging Face on first use, ~2.5 GB)."""
import urllib.request
from pathlib import Path

import pytest
from django.conf import settings

from copilot import audio, pipeline
from copilot.asr import NemotronASR, get_backend, set_backend
from copilot.models import Call

pytestmark = pytest.mark.gpu
SAMPLE_URL = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3"


@pytest.fixture(scope="module")
def sample_mp3():
    path = Path(settings.DATA_DIR) / "test-samples" / "obama.mp3"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SAMPLE_URL, path)
    return path.read_bytes()


@pytest.fixture(scope="module")
def nemotron():
    backend = NemotronASR(settings.ASR_MODEL_ID, "en-US", settings.ASR_DEVICE)
    backend.load()
    return backend


def test_model_runs_on_gpu(nemotron):
    import torch
    assert torch.cuda.is_available(), "no CUDA device visible to the container"
    assert nemotron.info()["device"].startswith("cuda")


def test_transcribes_real_speech(nemotron, sample_mp3):
    transcript = nemotron.transcribe(audio.decode(sample_mp3))
    text = transcript.text.lower()
    print("\nTRANSCRIPT:", transcript.text)
    assert transcript.duration > 5
    assert len(text.split()) > 20
    assert "america" in text or "american" in text


@pytest.mark.django_db
def test_full_pipeline_on_sample_call(nemotron):
    """tests/fixtures/sample_call.wav is a synthesized 20 s business call with one of each moment."""
    set_backend(nemotron)
    try:
        call = Call.objects.create(title="gpu e2e")
        pipeline.process_recording(call, [(Path(__file__).parent / "fixtures" / "sample_call.wav").read_bytes()])
        assert call.status == "done", call.error
        summary = pipeline.call_summary(call)
        print("\nTRANSCRIPT:", call.transcript, "\nSUMMARY:", summary)
        assert "annual plan" in summary["headline"]
        assert any(a["owner"] == "Sarah" for a in summary["action_items"])
        assert summary["counts"]["risk"] >= 1 and summary["counts"]["question"] >= 1
    finally:
        set_backend(None)


@pytest.fixture(scope="module")
def diarizer():
    from copilot.diarize import NemotronDiarizer
    d = NemotronDiarizer(settings.DIARIZER_MODEL_ID, settings.ASR_DEVICE, settings.DIARIZER_STREAMING_MODE)
    d.load()
    return d


@pytest.mark.django_db
@pytest.mark.parametrize("name,min_accuracy", [("quote_call_mono", 0.95), ("quote_call_phone_mono", 0.85)])
def test_mono_call_speakers_roles_and_use_cases(nemotron, diarizer, name, min_accuracy):
    """The main path: one mono recording -> diarize -> roles -> all three use cases. Prints role accuracy."""
    from copilot.diarize import set_diarizer
    from copilot.evaluation import load_truth, role_accuracy

    fixtures = Path(__file__).parent / "fixtures"
    set_backend(nemotron)
    set_diarizer(diarizer)
    try:
        call = Call.objects.create(title=name)
        pipeline.process_recording(call, [(fixtures / f"{name}.wav").read_bytes()])
        assert call.status == "done", call.error
        segs = [{"start": s.start, "end": s.end, "role": s.role} for s in call.segments.all()]
        for s, seg in zip(call.segments.all(), segs):
            print(f"\n{seg['start']:5.1f} {seg['role']:>7} ({s.speaker}): {s.text}", end="")
        score = role_accuracy(segs, load_truth(fixtures / f"{name}.truth.json"))
        print(f"\n{name}: roles={call.roles} accuracy={score['time_accuracy']} "
              f"turns={score['turns_correct']}/{score['turns']} covered={score['speech_covered']}")
        assert score["time_accuracy"] >= min_accuracy
        summary = pipeline.call_summary(call)
        assert summary["quotes"] and summary["quotes"][0]["icao"] == "KTEB"
        assert summary["dispute_warnings"]
        assert [a["speaker"] for a in summary["sentiment_alerts"]] == ["caller"]
    finally:
        set_backend(None)
        set_diarizer(None)


@pytest.mark.django_db
def test_two_leg_quote_call_hits_all_three_use_cases(nemotron):
    """tests/fixtures/quote_call_{agent,caller}.wav: two synthesized voices, one per leg, 44 s."""
    fixtures = Path(__file__).parent / "fixtures"
    set_backend(nemotron)
    try:
        call = Call.objects.create(title="gpu quote call")
        pipeline.process_recording(call, [(fixtures / "quote_call_agent.wav").read_bytes(),
                                          (fixtures / "quote_call_caller.wav").read_bytes()])
        assert call.status == "done", call.error
        for seg in call.segments.all():
            print(f"\n{seg.start:5.1f} {seg.speaker:>6}: {seg.text}", end="")
        summary = pipeline.call_summary(call)
        print("\nSUMMARY:", summary)
        assert {s.speaker for s in call.segments.all()} == {"agent", "caller"}
        assert summary["quotes"][0]["customer_number"] == "204511" and summary["quotes"][0]["icao"] == "KTEB"
        assert summary["quotes"][0]["offer_range_gppg"] == [0.085, 0.3125]
        assert summary["dispute_warnings"][0]["dispute_count"] == 7
        assert [a["speaker"] for a in summary["sentiment_alerts"]] == ["caller"]
        assert any(a["owner"] == "agent" and a["due"].lower() == "end of day" for a in summary["action_items"])
    finally:
        set_backend(None)


@pytest.mark.django_db
@pytest.mark.parametrize("name", ["quote_call_mono", "quote_call_phone_mono"])
def test_speech_service_mono_call(name):
    """The default stack: NeMo-Speech.cpp ASR + diarization in one request (needs the asr service running)."""
    import os
    import urllib.request

    from copilot.asr import NemoSpeechASR
    from copilot.diarize import FakeDiarizer, set_diarizer
    from copilot.evaluation import load_truth, role_accuracy

    url = os.environ.get("NEMO_SPEECH_URL", "http://asr:8080")
    try:
        urllib.request.urlopen(f"{url}/ready", timeout=3)
    except Exception:
        pytest.skip(f"speech service not reachable at {url}")
    fixtures = Path(__file__).parent / "fixtures"
    set_backend(NemoSpeechASR(url))
    set_diarizer(FakeDiarizer())  # enables the speakers path; the service does the diarizing
    try:
        call = Call.objects.create(title=name)
        pipeline.process_recording(call, [(fixtures / f"{name}.wav").read_bytes()])
        assert call.status == "done", call.error
        segs = [{"start": s.start, "end": s.end, "role": s.role} for s in call.segments.all()]
        score = role_accuracy(segs, load_truth(fixtures / f"{name}.truth.json"))
        print(f"\n{name} via speech service: roles={call.roles} accuracy={score['time_accuracy']} turns={score['turns_correct']}/{score['turns']}")
        assert score["time_accuracy"] >= 0.95
        summary = pipeline.call_summary(call)
        assert summary["quotes"] and summary["dispute_warnings"] and summary["sentiment_alerts"]
    finally:
        set_backend(None)
        set_diarizer(None)
