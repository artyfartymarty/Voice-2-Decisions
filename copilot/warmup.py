"""Load and exercise the in-process models when the web server starts, not on the first live call.

Without this the first live request after a restart waited ~10 s while Nemotron Diarization loaded and its CUDA
kernels compiled. Runs in a background thread so the server starts answering at once; a request that arrives
before warm-up ends just waits on the model's own load lock.
"""
import logging
import threading
import time

import numpy as np

log = logging.getLogger(__name__)
state = {"status": "not started", "seconds": None, "error": ""}


def _run():
    from .asr import get_backend
    from .audio import TARGET_SR
    from .diarize import get_diarizer

    t0 = time.perf_counter()
    state["status"] = "running"
    try:
        backend = get_backend()
        if hasattr(backend, "load"):
            backend.load()
        diarizer = get_diarizer()
        if diarizer is not None:
            diarizer.load()
            # one short streaming pass compiles the kernels the live path uses
            tone = (0.1 * np.sin(2 * np.pi * 220 * np.arange(3 * TARGET_SR) / TARGET_SR)).astype(np.float32)
            session = diarizer.session()
            session.push(tone)
            session.finish()
        state.update(status="done", seconds=round(time.perf_counter() - t0, 1))
        log.info("warm-up done in %.1f s", time.perf_counter() - t0)
    except Exception as exc:  # warm-up is best effort; the first request will load whatever is missing
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        log.exception("warm-up failed")


def start():
    if state["status"] == "not started":
        threading.Thread(target=_run, name="warmup", daemon=True).start()
