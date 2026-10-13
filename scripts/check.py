"""One command for "is everything OK?", with a PASS/FAIL line per step and exit code 1 on any failure.

    python scripts/check.py            # no GPU needed: settings, migrations, the fast test suite
    python scripts/check.py --stack    # on the running stack: also `manage.py doctor --golden` (real models)

In Docker:
    docker compose --profile test run --rm test python scripts/check.py
    docker compose exec web python scripts/check.py --stack
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(name, cmd, env=None):
    print(f"\n=== {name}: {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    code = subprocess.run(cmd, cwd=ROOT, env={**os.environ, **(env or {})}).returncode
    return name, code == 0, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stack", action="store_true", help="also run doctor --golden against the real models")
    a = ap.parse_args()
    py, manage = sys.executable, str(ROOT / "manage.py")
    # The code checks run on the fake model on purpose, so allow it there; doctor below uses the real settings.
    code_env = {"ALLOW_FAKE_BACKEND": "1"}
    steps = [
        run("settings", [py, manage, "check", "--fail-level", "WARNING"], code_env),
        run("migrations", [py, manage, "makemigrations", "--check", "--dry-run"], code_env),
        run("tests", [py, "-m", "pytest", "-q", "-rs"], {"ASR_BACKEND": "fake", **code_env}),
    ]
    if a.stack:
        steps.append(run("doctor", [py, manage, "doctor", "--golden"]))
    print("\n=== summary")
    for name, ok, secs in steps:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<12} {secs:5.1f} s")
    failed = [name for name, ok, _ in steps if not ok]
    print(f"\ncheck: {'FAILED: ' + ', '.join(failed) if failed else 'all passed'}"
          + ("" if a.stack else " (code only; add --stack on the GPU box to check the real models)"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
