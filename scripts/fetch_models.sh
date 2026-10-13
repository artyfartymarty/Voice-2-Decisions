#!/usr/bin/env bash
# Download the model files listed in models.lock into ./models, checking size and SHA-256.
# Files that already match are skipped; partial downloads resume.
#
#   scripts/fetch_models.sh            # from the pinned sources
#   MODEL_BASE_URL=https://my-bucket.s3.amazonaws.com/v2d-models scripts/fetch_models.sh
#   MODEL_URL_WINNOW_12B=https://... scripts/fetch_models.sh
#   MODELS_SOURCE=local scripts/fetch_models.sh   # never download: only verify files you copied into ./models
#
# Reads MODEL_BASE_URL / MODEL_URL_* from the environment or from .env.
set -euo pipefail
cd "$(dirname "$0")/.."
# .env fills in settings, but anything already set on the command line wins.
if [ -f .env ]; then
  while IFS='=' read -r key value; do
    case "$key" in MODELS_SOURCE|MODELS_DIR|MODEL_BASE_URL|MODEL_URL_*)
      [ -z "${!key+x}" ] && export "$key=$value" ;;
    esac
  done < <(grep -E '^[A-Z_]+=' .env)
fi
DEST="${MODELS_DIR:-./models}"
mkdir -p "$DEST"

fail=0
while IFS='|' read -r name path bytes sha url; do
  name=$(echo "$name" | xargs); path=$(echo "$path" | xargs); bytes=$(echo "$bytes" | xargs)
  sha=$(echo "$sha" | xargs); url=$(echo "$url" | xargs)
  case "$name" in ''|\#*) continue ;; esac
  override_var="MODEL_URL_${name}"
  if [ -n "${!override_var:-}" ]; then url="${!override_var}"
  elif [ -n "${MODEL_BASE_URL:-}" ]; then url="${MODEL_BASE_URL%/}/$path"; fi
  out="$DEST/$path"
  mkdir -p "$(dirname "$out")"
  if [ -f "$out" ] && [ "$(stat -c %s "$out")" = "$bytes" ] && echo "$sha  $out" | sha256sum -c --status; then
    echo "ok        $path"
    continue
  fi
  if [ "${MODELS_SOURCE:-download}" = "local" ]; then
    if [ -f "$out" ]; then echo "BAD       $path: present but size/SHA-256 differs from models.lock" >&2
    else echo "MISSING   $path  (expected $bytes bytes, sha256 $sha)" >&2; fi
    fail=1
    continue
  fi
  echo "download  $path  ($(numfmt --to=iec "$bytes" 2>/dev/null || echo "$bytes B"))  from ${url%%\?*}"
  curl -fL --retry 5 --retry-delay 3 -C - -o "$out.part" "$url"
  if [ "$(stat -c %s "$out.part")" != "$bytes" ] || ! echo "$sha  $out.part" | sha256sum -c --status; then
    echo "FAILED    $path: size or SHA-256 does not match models.lock (kept $out.part for a resume)" >&2
    fail=1
    continue
  fi
  mv "$out.part" "$out"
  echo "ok        $path"
done < models.lock

if [ "$fail" = 0 ]; then
  echo "all models present and verified in $DEST"
elif [ "${MODELS_SOURCE:-download}" = "local" ]; then
  echo "copy the files above into $DEST (see README: Use your own model files), then run this again" >&2; exit 1
else
  echo "some models failed; re-run to resume" >&2; exit 1
fi
