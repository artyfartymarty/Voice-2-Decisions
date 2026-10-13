"""Start winnow-server for a release preset, listening on all interfaces so the web container can reach it.

Winnow's own `scripts/winnow.py serve` resolves the pinned per-model settings but binds 127.0.0.1; this wrapper
reuses that resolution (serve_command) and adds --host. Mounted into the winnow-inference image by compose.

    python3 /wrap/serve.py --model e2b|q8 [--context 8192]
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path("/app")
sys.path.insert(0, str(ROOT / "scripts"))
import winnow  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model", default=os.environ.get("WINNOW_TARGET", "e2b"))
p.add_argument("--context", type=int, default=int(os.environ.get("WINNOW_CONTEXT", "8192")))
p.add_argument("--port", default=os.environ.get("WINNOW_PORT", "8091"))
a = p.parse_args()
ns = argparse.Namespace(model=a.model, reasoning="off", mtp="off", vision="off", context=a.context,
                        model_dir=Path(os.environ.get("WINNOW_MODEL_DIR", "/models")), server=None, cache=None)
args = winnow.serve_command(ns) + ["--host", "0.0.0.0", "--port", a.port]
if os.environ.get("WINNOW_API_KEY"):
    key = Path("/tmp/winnow.key")
    key.write_text(os.environ["WINNOW_API_KEY"])
    args += ["--api-key-file", str(key)]
print("winnow serve:", " ".join(args[1:]), flush=True)
os.execv(args[0], args)
