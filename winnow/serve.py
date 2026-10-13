"""Start winnow-server for a release preset, listening on all interfaces so the web container can reach it.

Winnow's own `scripts/winnow.py serve` resolves the pinned per-model settings but binds 127.0.0.1; this wrapper
reuses that resolution (serve_command) and adds --host. Mounted into the winnow-inference image by compose.

    python3 /wrap/serve.py --model e2b|q8 [--context N]

Context defaults: 12B (q8) runs at 4096. At 8192 its weights + F16 KV cache + buffers need ~17 GB, more than a
16 GB card has; Windows then silently spills to system RAM and prefill drops from ~35 ms to ~350 ms per request.
Sentiment requests are ~160 tokens, so 4K is plenty. E2B's pinned profiles are 8K or 64K, so it stays at 8192.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path("/app")
sys.path.insert(0, str(ROOT / "scripts"))
import winnow  # noqa: E402

DEFAULT_CONTEXT = {"q8": 4096, "e2b": 8192}

p = argparse.ArgumentParser()
p.add_argument("--model", default=os.environ.get("WINNOW_TARGET", "e2b"))
p.add_argument("--context", type=int, default=int(os.environ.get("WINNOW_CONTEXT") or 0))
p.add_argument("--port", default=os.environ.get("WINNOW_PORT", "8091"))
a = p.parse_args()
context = a.context or DEFAULT_CONTEXT.get(a.model, 8192)
cache = os.environ.get("WINNOW_CACHE") or None  # f16 (default) or q8_0 (less memory, measured ~25% slower)
ns = argparse.Namespace(model=a.model, reasoning="off", mtp="off", vision="off", context=context,
                        model_dir=Path(os.environ.get("WINNOW_MODEL_DIR", "/models")), server=None, cache=cache)
args = winnow.serve_command(ns) + ["--host", "0.0.0.0", "--port", a.port]
if os.environ.get("WINNOW_API_KEY"):
    key = Path("/tmp/winnow.key")
    key.write_text(os.environ["WINNOW_API_KEY"])
    args += ["--api-key-file", str(key)]
print("winnow serve:", " ".join(args[1:]), flush=True)
os.execv(args[0], args)
