#!/usr/bin/env bash
# Put Winnow-12B (Q8 GGUF, 12.7 GB) at models/winnow/gguf/Winnow-12B-Q8_0.gguf and check its size and SHA-256.
# A file that already matches is never re-fetched. MODELS_SOURCE picks where it comes from:
#
#   local   (default) never download: you copy the file in yourself (scp, aws s3 cp...); this only verifies it
#   url     one direct download from MODEL_URL (your S3 bucket, a presigned link, any web server), resumable
#   github  7 parts from a GitHub release (MODEL_REPO, MODEL_TAG), joined; release files must be under 2 GB,
#           and scripts/publish_model.py makes them
#
# Settings come from the environment or .env: MODELS_SOURCE, MODEL_URL, MODEL_REPO, MODEL_TAG.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f .env ]; then
  while IFS='=' read -r key value; do
    case "$key" in MODEL_REPO|MODEL_TAG|MODELS_SOURCE|MODEL_URL) [ -z "${!key+x}" ] && export "$key=$value" ;; esac
  done < <(grep -E '^[A-Z_]+=' .env)
fi
REPO="${MODEL_REPO:-artyfartymarty/Voice-2-Decisions}"
TAG="${MODEL_TAG:-models-v1}"
SOURCE="${MODELS_SOURCE:-local}"

NAME=Winnow-12B-Q8_0.gguf
BYTES=12669646592
SHA256=b710efc4c0d048ee61eed92c5fef5ce323a4d17e7c51f9f0533cc72ae50818ea
PART_BYTES=2000000000          # must match scripts/publish_model.py
OUT=models/winnow/gguf/$NAME
PARTS_DIR=models/winnow/gguf/.parts
mkdir -p "$(dirname "$OUT")"

verify() { [ -f "$1" ] && [ "$(stat -c %s "$1")" = "$BYTES" ] && echo "$SHA256  $1" | sha256sum -c --status; }

if [ -f "$OUT" ]; then
  echo "checking $OUT (SHA-256 of 12.7 GB takes about a minute)"
  if verify "$OUT"; then echo "ok        $OUT"; exit 0; fi
  echo "BAD       $OUT: size or SHA-256 differs; expected $BYTES bytes, sha256 $SHA256" >&2
  [ "$SOURCE" = "local" ] && exit 1
  mv "$OUT" "$OUT.bad"
fi
case "$SOURCE" in
  local)
    echo "MISSING   $OUT (expected $BYTES bytes, sha256 $SHA256)" >&2
    echo "copy Winnow-12B-Q8_0.gguf there (README: 'Get the model onto the instance'), or set MODELS_SOURCE=url / github in .env" >&2
    exit 1 ;;
  url)
    [ -n "${MODEL_URL:-}" ] || { echo "MODELS_SOURCE=url needs MODEL_URL in .env" >&2; exit 1; }
    echo "download  $NAME from ${MODEL_URL%%\?*}"
    curl -fL --retry 5 --retry-delay 3 -C - -o "$OUT.part" "$MODEL_URL"
    verify "$OUT.part" || { echo "FAILED    download does not match size/SHA-256 (kept $OUT.part; re-run to resume)" >&2; exit 1; }
    mv "$OUT.part" "$OUT"; rm -f "$OUT.bad"; echo "ok        $OUT (verified)"; exit 0 ;;
  github) ;;
  *) echo "MODELS_SOURCE must be local, url or github (got '$SOURCE')" >&2; exit 1 ;;
esac

count=$(( (BYTES + PART_BYTES - 1) / PART_BYTES ))
mkdir -p "$PARTS_DIR"
base="https://github.com/$REPO/releases/download/$TAG"
for i in $(seq 0 $((count - 1))); do
  part=$(printf '%s.part%02d' "$NAME" "$i")
  want=$(( i < count - 1 ? PART_BYTES : BYTES - PART_BYTES * (count - 1) ))
  if [ -f "$PARTS_DIR/$part" ] && [ "$(stat -c %s "$PARTS_DIR/$part")" = "$want" ]; then
    echo "ok        $part"; continue
  fi
  echo "download  $part ($((want / 1000000)) MB) from $base/$part"
  curl -fL --retry 5 --retry-delay 3 -C - -o "$PARTS_DIR/$part" "$base/$part" \
    || { echo "FAILED    $part: is the release published? See README, 'Put the model on GitHub'" >&2; exit 1; }
  [ "$(stat -c %s "$PARTS_DIR/$part")" = "$want" ] || { echo "FAILED    $part: expected $want bytes" >&2; exit 1; }
done

echo "joining $count parts -> $OUT"
cat $(for i in $(seq 0 $((count - 1))); do printf '%s/%s.part%02d ' "$PARTS_DIR" "$NAME" "$i"; done) > "$OUT.part"
if ! verify "$OUT.part"; then
  echo "FAILED    joined file does not match size/SHA-256; delete $PARTS_DIR and re-run" >&2; exit 1
fi
mv "$OUT.part" "$OUT"
rm -rf "$PARTS_DIR" "$OUT.bad"
echo "ok        $OUT (verified)"
