#!/bin/sh
# Download the pinned models on first start, then serve ASR + diarization.
set -eu
HF=https://huggingface.co

fetch() {  # repo file sha256 dest
  if [ -f "$4" ] && echo "$3  $4" | sha256sum -c - >/dev/null 2>&1; then
    return
  fi
  echo "downloading $2 ..."
  mkdir -p "$(dirname "$4")"
  curl -fsSL -C - -o "$4.part" "$HF/$1/resolve/main/$2"
  echo "$3  $4.part" | sha256sum -c -
  mv "$4.part" "$4"
}

ASR_GGUF="$MODELS_DIR/asr/nemotron-speech-streaming-en-0.6b.q8_0.gguf"
DIAR_GGUF="$MODELS_DIR/diar/Nemotron-3-Diarization.q8_0.gguf"
fetch nvidia/nemotron-speech-streaming-en-0.6b nemotron-speech-streaming-en-0.6b.q8_0.gguf \
  d9a01898d2a611c8764e23a1c2f45e70bbd5a425dc4de93692ac951dd603812d "$ASR_GGUF"
fetch nvidia/Nemotron-3-Diarization Nemotron-3-Diarization.q8_0.gguf \
  08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1 "$DIAR_GGUF"

set -- --host 0.0.0.0 --port 8080 --no-ui --asr-model "$ASR_GGUF" --diar-model "$DIAR_GGUF" "$@"
if [ -n "${NEMO_SPEECH_API_KEY:-}" ]; then
  export NEMO_SPEECH_HTTP_API_KEY="$NEMO_SPEECH_API_KEY"
fi
exec /opt/nemo-speech/bin/nemo-speech serve "$@"
