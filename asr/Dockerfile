# Speech service: NVIDIA NeMo-Speech.cpp v0.2.0 (prebuilt CUDA archive, SHA-pinned) serving
#   ASR          nemotron-speech-streaming-en-0.6b Q8 GGUF (English)
#   diarization  Nemotron-3-Diarization Q8 GGUF
# on :8080 with the OpenAI-style POST /v1/audio/transcriptions (verbose_json words, speakers, speech_contexts).
# Models are downloaded on first start into the models volume and checked against pinned SHA-256s.
FROM ubuntu:24.04

ARG NEMO_SPEECH_VERSION=0.2.0
ARG NEMO_SPEECH_SHA256=aef573fcc6c2c759401d2ed03e5ca9e401ed23b0e38cb852d917080c1e2d4515

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o /tmp/ns.tgz \
      https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/v${NEMO_SPEECH_VERSION}/nemo-speech-${NEMO_SPEECH_VERSION}-linux-x86_64-cuda.tar.gz \
 && echo "${NEMO_SPEECH_SHA256}  /tmp/ns.tgz" | sha256sum -c - \
 && mkdir -p /opt/nemo-speech \
 && tar -xzf /tmp/ns.tgz -C /opt/nemo-speech --strip-components=1 \
 && rm /tmp/ns.tgz

ENV LD_LIBRARY_PATH=/opt/nemo-speech/lib \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    MODELS_DIR=/models

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
EXPOSE 8080
HEALTHCHECK --interval=20s --timeout=5s --start-period=120s --retries=5 CMD curl -fs http://localhost:8080/ready || exit 1
ENTRYPOINT ["/entrypoint.sh"]
