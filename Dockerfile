# Text-to-decisions API: Python standard library only, no GPU.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PORT=8000
WORKDIR /app
COPY decisions decisions
COPY reference reference
COPY demos demos
# Only for the tests: the Winnow guard is a stdlib file, so its test runs here too (it runs in the Winnow image).
COPY winnow/guard.py winnow/guard.py

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz', timeout=2)" || exit 1
CMD ["python", "-m", "decisions.server"]
