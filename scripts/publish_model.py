"""Put Winnow-12B on GitHub: split the GGUF into parts under GitHub's 2 GB release-file limit and upload them as a
release on this repo. Run it once, wherever you already have the model file; scripts/fetch_model.sh then downloads
from GitHub only.

    python scripts/publish_model.py path/to/Winnow-12B-Q8_0.gguf
    python scripts/publish_model.py C:\\Users\\you\\models\\winnow\\gguf\\Winnow-12B-Q8_0.gguf --repo you/Voice-2-Decisions

Needs the GitHub CLI, logged in with push access (`gh auth login`). Works on Windows, macOS and Linux. Each part is
written to a temp folder, uploaded, then deleted, so it needs only ~2 GB of free disk. Re-running skips parts that
are already on the release with the right size.

Winnow-12B is Apache-2.0 (a Gemma 4 fine-tune by EldanRing); the release notes carry the license and source.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile

NAME = "Winnow-12B-Q8_0.gguf"
BYTES = 12669646592
SHA256 = "b710efc4c0d048ee61eed92c5fef5ce323a4d17e7c51f9f0533cc72ae50818ea"
PART_BYTES = 2_000_000_000  # must match scripts/fetch_model.sh
NOTES = (f"Winnow-12B Q8_0 GGUF ({BYTES} bytes, sha256 {SHA256}), split into {PART_BYTES}-byte parts for GitHub.\n\n"
         "scripts/fetch_model.sh downloads, joins and verifies them.\n\n"
         "Model: EldanRing/Winnow-12B, a fine-tune of Google Gemma 4 12B. License: Apache-2.0. See MODEL_LICENSES.md.")


def gh(*args, check=True, capture=False):
    return subprocess.run(["gh", *args], check=check, text=True, capture_output=capture)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", help=f"path to {NAME}")
    p.add_argument("--repo", default="artyfartymarty/Voice-2-Decisions")
    p.add_argument("--tag", default="models-v1")
    p.add_argument("--skip-verify", action="store_true", help="don't SHA-256 the file first (it takes about a minute)")
    a = p.parse_args()

    size = os.path.getsize(a.model)
    if size != BYTES:
        sys.exit(f"{a.model} is {size} bytes; Winnow-12B-Q8_0.gguf is {BYTES}")
    if not a.skip_verify:
        print("checking SHA-256 ...", flush=True)
        h = hashlib.sha256()
        with open(a.model, "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                h.update(block)
        if h.hexdigest() != SHA256:
            sys.exit(f"SHA-256 mismatch: got {h.hexdigest()}, expected {SHA256}")
        print("ok")

    if gh("release", "view", a.tag, "--repo", a.repo, check=False, capture=True).returncode != 0:
        gh("release", "create", a.tag, "--repo", a.repo, "--title", "Model files: Winnow-12B", "--notes", NOTES)
    have = {x["name"]: x["size"] for x in json.loads(
        gh("release", "view", a.tag, "--repo", a.repo, "--json", "assets", capture=True).stdout)["assets"]}

    count = -(-BYTES // PART_BYTES)
    with open(a.model, "rb") as src, tempfile.TemporaryDirectory() as tmp:
        for i in range(count):
            part = f"{NAME}.part{i:02d}"
            want = min(PART_BYTES, BYTES - i * PART_BYTES)
            if have.get(part) == want:
                print(f"ok        {part} already on the release")
                continue
            src.seek(i * PART_BYTES)
            path = os.path.join(tmp, part)
            with open(path, "wb") as out:
                left = want
                while left:
                    chunk = src.read(min(left, 1 << 24))
                    out.write(chunk)
                    left -= len(chunk)
            print(f"upload    {part} ({want // 1_000_000} MB)", flush=True)
            gh("release", "upload", a.tag, path, "--repo", a.repo, "--clobber")
            os.remove(path)
    print(f"done: https://github.com/{a.repo}/releases/tag/{a.tag}")


if __name__ == "__main__":
    main()
