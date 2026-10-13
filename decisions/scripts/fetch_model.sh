#!/usr/bin/env bash
# Get Winnow-12B (Q8 GGUF, 12.7 GB) from this repo's GitHub release into models/winnow/gguf/, checked by SHA-256.
#
# GitHub release files must be under 2 GB, so the model is published as 7 parts
# (scripts/publish_model.py makes them). This downloads each part (resuming partial downloads), joins them,
# and checks the joined file against the size and SHA-256 below. A file that already matches is never re-fetched.
#
#   scripts/fetch_model.sh                          # from github.com/$MODEL_REPO/releases/tag/$MODEL_TAG
#   MODELS_SOURCE=local scripts/fetch_model.sh      # never download; only verify a file you copied in
#
# Settings come from the environment or .env: MODEL_REPO, MODEL_TAG, MODELS_SOURCE.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f .env ]; then
  while IFS='=' read -r key value; do
    case "$key" in MODEL_REPO|MODEL_TAG|MODELS_SOURCE) [ -z "${!key+x}" ] && export "$key=$value" ;; esac
  done < <(grep -E '^[A-Z_]+=' .env)
fi
REPO="${MODEL_REPO:-artyfartymarty/Voice-2-Decisions}"
TAG="${MODEL_TAG:-models-v1}"

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
  [ "${MODELS_SOURCE:-download}" = "local" ] && exit 1
  mv "$OUT" "$OUT.bad"
fi
if [ "${MODELS_SOURCE:-download}" = "local" ]; then
  echo "MISSING   $OUT (expected $BYTES bytes, sha256 $SHA256)" >&2; exit 1
fi

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
