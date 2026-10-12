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
# ASR_BACKEND: "nemotron" (real model, GPU recommended) or "fake" (no model; used by tests and CPU-only boxes).
ASR_BACKEND = os.environ.get("ASR_BACKEND", "nemotron")
ASR_MODEL_ID = os.environ.get("ASR_MODEL_ID", "nvidia/nemotron-3.5-asr-streaming-0.6b")
ASR_LANGUAGE = os.environ.get("ASR_LANGUAGE", "en-US")  # a locale such as en-US, or "auto"
ASR_DEVICE = os.environ.get("ASR_DEVICE", "auto")  # auto | cuda | cpu
ASR_CHUNK_SECONDS = float(os.environ.get("ASR_CHUNK_SECONDS", "30"))
FAKE_ASR_TEXT = os.environ.get("FAKE_ASR_TEXT", "")

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"console": {"class": "logging.StreamHandler"}},
    "root": {"handlers": ["console"], "level": os.environ.get("LOG_LEVEL", "INFO")},
}
