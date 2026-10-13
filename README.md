# Voice-2-Decisions

> **Speech handled upstream (Amazon Nova Sonic)?** Use [`decisions/`](decisions/README.md): transcript text in,
> variables parsed, functions auto-called, Winnow-12B surfaces the result the caller asked for. Two containers;
> you copy the model file onto the instance.

A live call copilot. A mono call recording or stream goes in; it works out who's speaking (agent / caller), transcribes,
and fires three use cases while the call is happening:

| Category | What the call handler gets | How |
|---|---|---|
| GP | Customer asks for a quote: addressable volume at that location, the margin range to come down within, dispatcher alerted | static lookup on customer + location |
| Avoiding Disputes | Quoting a dispute-prone location or customer: restate the fees, plus wording for the written quote | keyword + notes / dispute list |
| Customer Experience | Caller or agent turning negative: on-call prompt, managers alerted in real time | **Winnow-12B** sentiment + alert routing |

Three containers on one NVIDIA GPU: **web** (Django UI + API, :8080), **asr** (NVIDIA NeMo-Speech.cpp: English
Nemotron ASR + Nemotron diarization), **winnow** (Winnow-12B). Deeper notes, measured results and troubleshooting are in
[HANDOFF.md](HANDOFF.md).

## Quick start on AWS (about five commands)

**Instance:** `g5.2xlarge` (A10G 24 GB) · **AMI:** *Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)* ·
**Disk:** 100 GB · **Security group:** SSH only.

```bash
# 1. on the instance
git clone https://github.com/artyfartymarty/Voice-2-Decisions.git && cd Voice-2-Decisions
# 2. check GPU + driver, write .env, get models (~14 GB), build, start, run the checks
./scripts/setup.sh
```

The repo is private: for the clone, use `gh auth login` first or
`git clone https://<your-github-token>@github.com/artyfartymarty/Voice-2-Decisions.git`.

```bash
# 3. on your laptop: tunnel to the UI
ssh -L 8080:localhost:8080 ubuntu@<instance-public-dns>
# 4. open http://localhost:8080 and click a test call (or drop your own WAV); it streams like a live call
```

`setup.sh` ends with `check.py --stack`, which must print `check: all passed`. The first run takes a while: the model
download, plus compiling Winnow for your GPU (`CUDA_ARCH` is set automatically from `nvidia-smi`).

## Test with a WAV file

- **In the browser:** the start page lists `tests/fixtures/*.wav`. Click one, or drop a WAV, and it plays and streams in
  real time while the transcript, prompts and alerts fill in.
- **From the instance:**

```bash
docker compose exec web python scripts/stream_wav.py tests/fixtures/quote_call_phone_mono.wav --chunk 2
# copy your own: scp call.wav ubuntu@<instance>:Voice-2-Decisions/  then
docker compose cp call.wav web:/tmp/call.wav && docker compose exec web python scripts/stream_wav.py /tmp/call.wav
```

It prints per-chunk latency and when each use case fired. Expect ~0.1–0.2 s per 2 s chunk.

## Use your own model files instead of downloading

By default `scripts/fetch_models.sh` (run by `setup.sh`) downloads the weights from their publishers (the weights are
only published on Hugging Face; the runtimes come from GitHub). To use files you copy in yourself, put them in this
exact layout under `./models` and tell the script not to download:

```
models/
├── winnow/gguf/Winnow-12B-Q8_0.gguf                     12669646592 bytes  sha256 b710efc4c0d048ee61eed92c5fef5ce323a4d17e7c51f9f0533cc72ae50818ea
├── asr/nemotron-speech-streaming-en-0.6b.q8_0.gguf        699872960 bytes  sha256 d9a01898d2a611c8764e23a1c2f45e70bbd5a425dc4de93692ac951dd603812d
├── diar/Nemotron-3-Diarization.q8_0.gguf                  107012128 bytes  sha256 08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1
└── diar-hf/
    ├── model.safetensors                                  396954592 bytes  sha256 c074d86335b3b794f8fa5edc25594558f128bdb3914d27806a3a5a2e44963cb6
    ├── config.json                                             1458 bytes  sha256 dd795ec07ffa09acf18d231aa3fa8642aa5c521760953754e02559a7413b6873
    └── processor_config.json                                    623 bytes  sha256 91726746788a024a2f0f2f699a6c3667f0d74e6a0e9288fc44fe3018e23af68e
```

**1. Get the files onto the instance**, any way you like:

```bash
# from your laptop (the folder you have locally, in the layout above)
scp -r models ubuntu@<instance>:Voice-2-Decisions/
# or from an S3 bucket laid out the same way
aws s3 cp --recursive s3://<bucket>/v2d-models/ models/
```

**2. Switch off downloading** in `.env`:

```
MODELS_SOURCE=local
```

**3. Run `./scripts/setup.sh` as usual.** With `MODELS_SOURCE=local` it never downloads. It checks each file's
size and SHA-256 against `models.lock`, prints `MISSING` or `BAD` for anything wrong (with the expected size and hash),
and stops before building. Check by hand any time with `MODELS_SOURCE=local scripts/fetch_models.sh`.

Other ways to point at your files:
- **A mirror URL for everything:** `MODEL_BASE_URL=https://<bucket>.s3.amazonaws.com/v2d-models`. Each file is
  fetched from `<base>/<path above>` and checksummed.
- **One file elsewhere:** `MODEL_URL_WINNOW_12B=https://...` (also `MODEL_URL_ASR_EN`, `MODEL_URL_DIAR_GGUF`,
  `MODEL_URL_DIAR_HF_WEIGHTS`, `MODEL_URL_DIAR_HF_CONFIG`, `MODEL_URL_DIAR_HF_PROCESSOR`).
- **Files already present and correct** are always reused, in every mode: only missing or bad files are fetched.

The pinned sources, sizes and hashes live in [`models.lock`](models.lock); licenses are in
[MODEL_LICENSES.md](MODEL_LICENSES.md) (Apache-2.0, NVIDIA Open Model License, OpenMDW-1.1; all allow redistribution
with the license attached).

## Everyday commands

```bash
docker compose ps                                   # status
docker compose logs -f web asr winnow               # logs
docker compose exec web python scripts/check.py --stack   # health check on the real models
docker compose --profile test run --rm test         # fast test suite (no GPU)
./scripts/setup.sh --skip-build                     # after a reboot: verify models, start, check
docker compose down                                 # stop
```

Real reference data (customers, location notes) instead of the synthetic samples: see HANDOFF.md, step 3.
