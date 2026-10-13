#!/bin/sh
# Serve ASR + diarization from the model files mounted at /models (scripts/fetch_models.sh puts them there).
set -eu
ASR_GGUF="$MODELS_DIR/asr/nemotron-speech-streaming-en-0.6b.q8_0.gguf"
DIAR_GGUF="$MODELS_DIR/diar/Nemotron-3-Diarization.q8_0.gguf"
for f in "$ASR_GGUF" "$DIAR_GGUF"; do
  if [ ! -f "$f" ]; then
    echo "missing $f: run scripts/fetch_models.sh on the host first" >&2
    exit 1
  fi
done

set -- --host 0.0.0.0 --port 8080 --no-ui --asr-model "$ASR_GGUF" --diar-model "$DIAR_GGUF" "$@"
if [ -n "${NEMO_SPEECH_API_KEY:-}" ]; then
  export NEMO_SPEECH_HTTP_API_KEY="$NEMO_SPEECH_API_KEY"
fi
exec /opt/nemo-speech/bin/nemo-speech serve "$@"
