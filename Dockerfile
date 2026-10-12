# Voice-2-Decisions: Django web UI + Nemotron ASR + winnow.
# Base ships Python 3.12 and torch 2.11 built for CUDA 12.8 (covers Ampere through Blackwell GPUs,
# including AWS g5/g6/g6e and RTX 50-series).
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    DATA_DIR=/data \
    HF_HOME=/data/hf

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libsndfile1 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements.txt -r requirements-dev.txt

COPY . .
RUN mkdir -p /data

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD curl -fs http://localhost:8000/healthz || exit 1

# One worker so the model is loaded once; threads share it (ASR calls are serialized by a lock).
CMD ["sh", "-c", "python manage.py migrate --noinput && exec gunicorn v2d.wsgi:application --bind 0.0.0.0:8000 --workers 1 --threads 4 --timeout 900"]
