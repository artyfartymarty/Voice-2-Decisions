from django.conf import settings


def mode(request):
    """Which speech backend is serving, for the header badge on every page (demo / fake / nemotron / nemo_speech)."""
    return {"asr_backend": settings.ASR_BACKEND}
