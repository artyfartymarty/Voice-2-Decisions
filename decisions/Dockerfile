# Text-to-decisions API: Python standard library only, no GPU. Built from the repo root (see docker-compose.yml)
# so it can reuse the extractor in copilot/ and the reference data in fixtures/reference.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PORT=8000
WORKDIR /app
COPY copilot/__init__.py copilot/extract.py copilot/slots.py copilot/reference.py copilot/
COPY fixtures/reference fixtures/reference
COPY tests/fixtures/*.script.json tests/fixtures/
COPY decisions decisions

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz', timeout=2)" || exit 1
CMD ["python", "-m", "decisions.server"]
