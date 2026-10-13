"""Django settings. Everything deploy-specific comes from environment variables (see .env.example)."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)


def env_bool(name, default=False):
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-only-insecure-key-change-me")
DEBUG = env_bool("DJANGO_DEBUG", False)
ALLOWED_HOSTS = [h.strip() for h in os.environ.get("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",") if h.strip()]
CSRF_TRUSTED_ORIGINS = [o.strip() for o in os.environ.get("DJANGO_CSRF_TRUSTED_ORIGINS", "").split(",") if o.strip()]

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.staticfiles",
    "copilot",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
]

ROOT_URLCONF = "v2d.urls"
WSGI_APPLICATION = "v2d.wsgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {"context_processors": ["django.template.context_processors.request"]},
    }
]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": os.environ.get("SQLITE_PATH", str(DATA_DIR / "db.sqlite3")),
    }
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
USE_TZ = True
TIME_ZONE = "UTC"
STATIC_URL = "static/"

# Uploads: calls can be long, so allow large audio files.
DATA_UPLOAD_MAX_MEMORY_SIZE = int(os.environ.get("MAX_UPLOAD_MB", "200")) * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = DATA_UPLOAD_MAX_MEMORY_SIZE

# --- Voice-2-Decisions ---
# ASR_BACKEND:
#   nemo_speech  the speech service (NeMo-Speech.cpp, English Nemotron Q8 + diarization) at NEMO_SPEECH_URL (fastest)
#   nemotron     Nemotron 3.5 ASR in this process via Transformers (no extra service; GPU recommended)
#   fake         no model; tests and CPU-only boxes
ASR_BACKEND = os.environ.get("ASR_BACKEND", "nemotron")
NEMO_SPEECH_URL = os.environ.get("NEMO_SPEECH_URL", "http://asr:8080")
NEMO_SPEECH_API_KEY = os.environ.get("NEMO_SPEECH_API_KEY", "")
ASR_WORD_BOOST = float(os.environ.get("ASR_WORD_BOOST", "4.0"))  # boost for customer / airport / FBO names; 0 = off
ASR_MODEL_ID = os.environ.get("ASR_MODEL_ID", "nvidia/nemotron-3.5-asr-streaming-0.6b")
ASR_LANGUAGE = os.environ.get("ASR_LANGUAGE", "en-US")  # a locale such as en-US, or "auto"
ASR_DEVICE = os.environ.get("ASR_DEVICE", "auto")  # auto | cuda | cpu
ASR_CHUNK_SECONDS = float(os.environ.get("ASR_CHUNK_SECONDS", "30"))
FAKE_ASR_TEXT = os.environ.get("FAKE_ASR_TEXT", "")
# Mono calls: who spoke when. nemotron = nvidia/Nemotron-3-Diarization, fake = tests, off = no speaker labels.
# With ASR_BACKEND=nemo_speech, recordings get speakers from the speech service in the same request; this setting
# then only drives live calls (whose streaming session keeps speaker IDs stable across chunks).
DIARIZER = os.environ.get("DIARIZER", "nemotron" if ASR_BACKEND in ("nemotron", "nemo_speech") else "off")
DIARIZER_MODEL_ID = os.environ.get("DIARIZER_MODEL_ID", "nvidia/Nemotron-3-Diarization")
DIARIZER_STREAMING_MODE = os.environ.get("DIARIZER_STREAMING_MODE", "low_latency")  # or very_low_latency, ultra_low_latency
# Live calls: audio held back mid-word is transcribed anyway after this many seconds without a new chunk (0 = off).
LIVE_FLUSH_S = float(os.environ.get("LIVE_FLUSH_S", "0.9"))
# Two-leg calls: "separate" transcribes each leg with a speaker label (default); "mix" folds them to one mono leg.
LEG_MODE = os.environ.get("LEG_MODE", "separate")

# Reference data for the use-case tools (customers.csv, location_notes.csv). Defaults to the synthetic fixtures.
REFERENCE_DIR = Path(os.environ.get("REFERENCE_DIR") or BASE_DIR / "fixtures" / "reference")
# Sentiment alert: fire when the average of the last SENTIMENT_WINDOW scored sentences drops to SENTIMENT_ALERT_AT or below.
SENTIMENT_WINDOW = int(os.environ.get("SENTIMENT_WINDOW", "3"))
SENTIMENT_ALERT_AT = float(os.environ.get("SENTIMENT_ALERT_AT", "-0.4"))
# Customer Experience sentiment: winnow (POST {WINNOW_URL}/v1/systemone), lexicon, or auto (winnow if WINNOW_URL is set).
SENTIMENT_BACKEND = os.environ.get("SENTIMENT_BACKEND", "auto")
WINNOW_URL = os.environ.get("WINNOW_URL", "")
WINNOW_API_KEY = os.environ.get("WINNOW_API_KEY", "")
# Designated managers / leads for real-time sentiment alerts: comma-separated webhook URLs (falls back to NOTIFY_WEBHOOK_URL).
MANAGER_WEBHOOKS = [u.strip() for u in os.environ.get("MANAGER_WEBHOOKS", "").split(",") if u.strip()]
# GP: flag the margin as high when CURRENT_GPPG is at least this many times MIN_GPPG.
MARGIN_HIGH_RATIO = float(os.environ.get("MARGIN_HIGH_RATIO", "3.0"))
# GP: addressable volume below this share of total P12M volume is flagged for Yield to verify.
ADDRESSABLE_CHECK_RATIO = float(os.environ.get("ADDRESSABLE_CHECK_RATIO", "0.5"))
# validate_tail_number fires only for tail numbers extracted with at least this confidence.
TAIL_MIN_CONFIDENCE = float(os.environ.get("TAIL_MIN_CONFIDENCE", "0.8"))
# Dispute guidance fires for locations with at least this many disputes in their private notes.
DISPUTE_MIN_COUNT = int(os.environ.get("DISPUTE_MIN_COUNT", "3"))
# Test page: pregenerated WAV calls listed on the entry page (click one to stream it as a simulated live call).
TEST_CALLS_DIR = Path(os.environ.get("TEST_CALLS_DIR") or BASE_DIR / "tests" / "fixtures")
# Optional: a WebSocket that announces call updates (the gateway's events stream). When set, the call page and the
# alert view refresh as soon as a message arrives; otherwise they poll every LIVE_REFRESH_MS while a call is live.
GATEWAY_WS_URL = os.environ.get("GATEWAY_WS_URL", "")
LIVE_REFRESH_MS = int(os.environ.get("LIVE_REFRESH_MS", "1500"))
# Optional: alerts (dispatcher, managers) are POSTed here as JSON, e.g. a Slack or Teams incoming webhook.
NOTIFY_WEBHOOK_URL = os.environ.get("NOTIFY_WEBHOOK_URL", "")

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"console": {"class": "logging.StreamHandler"}},
    "root": {"handlers": ["console"], "level": os.environ.get("LOG_LEVEL", "INFO")},
}
