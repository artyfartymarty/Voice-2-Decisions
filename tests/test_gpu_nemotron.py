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
