"""Is this deployment real and working? One line per check, PASS / WARN / FAIL, exit code 1 on any FAIL.

    python manage.py doctor              # config, database, reference data, GPU, services, a speech probe
    python manage.py doctor --golden     # also stream the golden calls through the real models (GPU box)
    python manage.py doctor --no-probe   # skip loading the speech model
    python manage.py doctor --json

The speech probe transcribes two different turns of a test call. It fails when the text is empty, identical for
both turns, or the fake model's canned paragraph, which is exactly what a misconfigured server looks like.
"""
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
PROBE_WAV = "tests/fixtures/quote_call_mono.wav"
MIN_DRIVER = 580  # the Winnow image is built on CUDA 13.0 (HANDOFF.md, step 0)


def _get_json(url: str, timeout: float = 3.0) -> dict:
    import urllib.request
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


class Command(BaseCommand):
    help = "Check config, database, reference data, GPU, speech and sentiment services, and probe the speech model."
    requires_system_checks = []  # report a bad config here instead of refusing to run

    def add_arguments(self, parser):
        parser.add_argument("--no-probe", action="store_true", help="don't load or call the speech model")
        parser.add_argument("--golden", action="store_true", help="stream the golden calls through the real models")
        parser.add_argument("--json", action="store_true", help="print the results as JSON")

    def handle(self, *args, **opts):
        self.results = []
        checks = [self.check_config, self.check_secret, self.check_database, self.check_reference, self.check_version,
                  self.check_gpu, self.check_speech_service, self.check_sentiment]
        if not opts["no_probe"]:
            checks.append(self.check_probe)
        if opts["golden"]:
            checks.append(self.check_golden)
        for check in checks:
            try:
                check()
            except Exception as exc:
                self.add(FAIL, check.__name__.removeprefix("check_"), f"{type(exc).__name__}: {exc}")
        failed = [r for r in self.results if r["status"] == FAIL]
        if opts["json"]:
            self.stdout.write(json.dumps({"ok": not failed, "checks": self.results}, indent=2))
        else:
            style = {PASS: self.style.SUCCESS, WARN: self.style.WARNING, FAIL: self.style.ERROR}
            for r in self.results:
                self.stdout.write(f"{style[r['status']](r['status'])}  {r['name']:<16} {r['detail']}")
            verdict = f"{len(failed)} check(s) FAILED" if failed else "all checks passed"
            self.stdout.write((self.style.ERROR if failed else self.style.SUCCESS)(f"\ndoctor: {verdict}"))
        if failed:
            raise SystemExit(1)

    def add(self, status, name, detail):
        self.results.append({"status": status, "name": name, "detail": detail})

    # --- checks ---

    def check_config(self):
        from copilot import guards
        problems = guards.config_problems()
        for p in problems:
            self.add(FAIL, "config", p)
        if not problems:
            used = f"ASR_BACKEND={settings.ASR_BACKEND}, DIARIZER={settings.DIARIZER}, SENTIMENT_BACKEND={settings.SENTIMENT_BACKEND}"
            if guards.fake_backends():
                self.add(WARN, "config", f"{used}; fake backend allowed by ALLOW_FAKE_BACKEND=1, transcripts are canned")
            else:
                self.add(PASS, "config", used)

    def check_secret(self):
        if settings.SECRET_KEY in ("dev-only-insecure-key-change-me", "change-me", "") and not settings.DEBUG:
            self.add(FAIL, "secret key", "DJANGO_SECRET_KEY is the placeholder; generate one (HANDOFF.md, step 3)")
        else:
            self.add(PASS, "secret key", f"set; DEBUG={settings.DEBUG}, ALLOWED_HOSTS={','.join(settings.ALLOWED_HOSTS)}")

    def check_database(self):
        from django.db import connection
        from django.db.migrations.executor import MigrationExecutor
        pending = MigrationExecutor(connection).migration_plan(MigrationExecutor(connection).loader.graph.leaf_nodes())
        if pending:
            self.add(FAIL, "database", f"{len(pending)} migration(s) not applied: run python manage.py migrate")
        else:
            self.add(PASS, "database", f"{connection.vendor}, migrations applied")

    def check_reference(self):
        from copilot import extract, reference
        ref = reference.get()
        n_cust = len(extract.get().customers)
        if not n_cust or not ref.locations:
            self.add(FAIL, "reference data", f"{settings.REFERENCE_DIR}: {n_cust} customers, {len(ref.locations)} locations")
            return
        synthetic = Path(settings.REFERENCE_DIR).resolve() == (Path(settings.BASE_DIR) / "fixtures" / "reference").resolve()
        self.add(WARN if synthetic else PASS, "reference data",
                 f"{n_cust} customers, {len(ref.locations)} locations from {settings.REFERENCE_DIR}"
                 + (" (the synthetic samples, not real exports)" if synthetic else ""))

    def check_version(self):
        import os
        sha = os.environ.get("GIT_SHA", "")
        if not sha and shutil.which("git") and (Path(settings.BASE_DIR) / ".git").exists():
            run = lambda *a: subprocess.run(["git", *a], cwd=settings.BASE_DIR, capture_output=True, text=True).stdout.strip()
            sha = run("rev-parse", "--short", "HEAD") + (" + uncommitted changes" if run("status", "--porcelain") else "")
        self.add(PASS, "code version", sha or "unknown (no .git here; pass GIT_SHA to the container to see it)")

    def check_gpu(self):
        needs_gpu = settings.ASR_BACKEND == "nemotron" or settings.DIARIZER == "nemotron"
        try:
            import torch
            cuda = torch.cuda.is_available()
            name = torch.cuda.get_device_name(0) if cuda else ""
        except ImportError:
            cuda, name = False, "torch not installed"
        if cuda:
            self.add(PASS, "gpu", name)
        elif needs_gpu and settings.ASR_DEVICE != "cpu":
            self.add(FAIL, "gpu", "no CUDA in this process, but DIARIZER/ASR_BACKEND=nemotron run models here. "
                                  "Start with -f docker-compose.gpu.yml (or set ASR_DEVICE=cpu on purpose)")
        else:
            self.add(WARN if needs_gpu else PASS, "gpu", f"no CUDA here{f' ({name})' if name else ''}; "
                     + ("models will run on the CPU (slow)" if needs_gpu else "not needed by this process"))
        if shutil.which("nvidia-smi"):
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=20).stdout.strip()
            m = re.search(r"(\d+)\.\d+", out.split(",")[1]) if "," in out else None
            if m and int(m.group(1)) < MIN_DRIVER:
                self.add(FAIL, "gpu driver", f"{out}: driver {m.group(1)} is older than {MIN_DRIVER} (Winnow needs CUDA 13)")
            else:
                self.add(PASS if m else WARN, "gpu driver", out or "nvidia-smi gave no answer")

    def check_speech_service(self):
        if settings.ASR_BACKEND != "nemo_speech":
            return
        url = settings.NEMO_SPEECH_URL.rstrip("/")
        try:
            ready = _get_json(f"{url}/ready")
        except Exception as exc:
            self.add(FAIL, "speech service", f"{url}/ready unreachable: {exc}")
            return
        caps = ready.get("capabilities", [])
        if not ready.get("ready"):
            self.add(FAIL, "speech service", f"{url} answered but is not ready: {ready}")
        elif settings.DIARIZER != "off" and caps and "diarization" not in caps:
            self.add(WARN, "speech service", f"ready, but without diarization (capabilities: {caps})")
        else:
            self.add(PASS, "speech service", f"{url} ready, capabilities: {caps or 'not reported'}")

    def check_sentiment(self):
        backend = settings.SENTIMENT_BACKEND
        if backend == "auto":
            backend = "winnow" if settings.WINNOW_URL else "lexicon"
        if backend != "winnow":
            self.add(WARN, "sentiment", "word list only (Winnow off): it catches about 1 in 10 negative lines")
            return
        url = settings.WINNOW_URL.rstrip("/")
        try:
            health = _get_json(f"{url}/health")
            ok = health.get("status") == "ok"
            self.add(PASS if ok else WARN, "sentiment", f"Winnow at {url}: {health}")
        except Exception as exc:
            self.add(WARN, "sentiment", f"Winnow at {url} unreachable ({exc}); alerts fall back to the word list")

    def check_probe(self):
        from copilot import audio
        from copilot.asr import FakeASR, get_backend
        from copilot.evaluation import load_truth

        wav = Path(settings.BASE_DIR) / PROBE_WAV
        samples = audio.decode(wav.read_bytes())
        turns = load_truth(wav.with_suffix(".truth.json"))[:2]
        backend = get_backend()
        texts, t0 = [], time.perf_counter()
        for turn in turns:
            span = samples[int(turn["start"] * audio.TARGET_SR): int(turn["end"] * audio.TARGET_SR)]
            texts.append(backend.transcribe(span).text.strip())
        ms = round(1000 * (time.perf_counter() - t0))
        heard = " | ".join(f'"{t[:60]}"' for t in texts)
        canned = {FakeASR.DEFAULT_TEXT, settings.FAKE_ASR_TEXT} - {""}
        if backend.name == "fake" or any(t in canned for t in texts):
            self.add(FAIL, "speech probe", f"the fake model answered ({backend.name}): {heard}")
        elif not all(texts):
            self.add(FAIL, "speech probe", f"{backend.name} returned no text for speech: {heard}")
        elif len(set(texts)) < len(texts):
            self.add(FAIL, "speech probe", f"{backend.name} returned the same text for two different turns: {heard}")
        else:
            self.add(PASS if ms < 3000 else WARN, "speech probe", f"{backend.name}, {ms} ms for 2 turns: {heard}")

    def check_golden(self):
        from copilot import golden
        for case in golden.load():
            if "wav" not in case:
                continue
            t0 = time.perf_counter()
            result = golden.play_wav(case)
            failures = golden.compare(case, result)
            took = f"{time.perf_counter() - t0:.1f} s"
            if failures:
                self.add(FAIL, f"golden {case['name']}", f"{took}; " + "; ".join(failures))
                for line in result["transcript"][:6]:
                    self.add(WARN, "  heard", line[:120])
            else:
                fired = ", ".join(t["tool"] for t in result["tools"] if t["tool"] in {e["tool"] for e in case["expect"]})
                self.add(PASS, f"golden {case['name']}", f"{took}; fired {fired}")
