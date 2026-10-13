# Handoff: run Voice-2-Decisions in a container on EC2 and test it

These are instructions for the person or agent setting this up. Follow the steps in order. Every step lists the
command and what success looks like. All of it was verified on the author's machine (RTX 5070 Ti, Docker Desktop)
on 2026-10-12: 35 fast tests passed, 3 GPU tests passed, and the smoke test passed.

## 0. What you need

- **EC2 instance with an NVIDIA GPU.** A `g5.xlarge` (A10G 24 GB) or `g6.xlarge` (L4 24 GB) is plenty, since the
  model uses about 1.6 GB of VRAM.
- **AMI:** "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04 or 24.04)". It ships the NVIDIA driver, Docker and
  the NVIDIA Container Toolkit. With another AMI, install those three yourself. The driver must be **570 or newer**,
  because the image uses CUDA 12.8.
- **Disk:** at least 40 GB on the root volume (image about 12 GB, model cache about 2.5 GB).
- **Outbound HTTPS** to Docker Hub, PyPI and huggingface.co, used during build and the first model load.
- **Inbound:** SSH. For the web UI, either open TCP 8000 to your IP or use an SSH tunnel (step 6).
- **No GPU?** Everything except the real-model tests still runs on a CPU instance. Skip the `-f docker-compose.gpu.yml`
  parts and set `ASR_BACKEND=fake` in `.env`.

## 1. Get the code onto the instance

The repo is `https://github.com/artyfartymarty/Voice-2-Decisions`:

```bash
git clone https://github.com/artyfartymarty/Voice-2-Decisions.git
cd Voice-2-Decisions
```

If you received the files directly instead, as plain files and not a zip, copy the whole folder up with
`scp -r` or `rsync` and `cd` into it.

## 2. Check the GPU is visible to Docker

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

Success means both print the GPU table. If the second fails, the NVIDIA Container Toolkit is missing. Install it
(`sudo apt-get install -y nvidia-container-toolkit && sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`).

## 3. Configure

```bash
cp .env.example .env
sed -i "s|^DJANGO_SECRET_KEY=.*|DJANGO_SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_urlsafe(50))')|" .env
```

To reach the UI by the instance's address instead of a tunnel, also edit `.env`:

```
DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1,<public-dns-or-ip>
DJANGO_CSRF_TRUSTED_ORIGINS=http://<public-dns-or-ip>:8000
```

## 4. Build and start

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
curl -s "http://localhost:8000/healthz?load=1"
```

The first `healthz?load=1` downloads the model, which takes a minute or two. Success looks like:
`"ok": true ... "backend": "nemotron" ... "loaded": true, "cuda": true, "device": "cuda:0"`.

## 5. Run the tests

```bash
# a) Fast suite: rules, audio, API, tools. No GPU or model needed. Expect: 35 passed, 3 skipped.
docker compose --profile test run --rm test

# b) Real-model suite on the GPU. Expect: 3 passed.
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile test run --rm -e RUN_GPU_TESTS=1 test \
  pytest -v -rs tests/test_gpu_nemotron.py

# c) End to end against the running server: upload, summary, and a live call in 5 s chunks. Expect: SMOKE TEST PASSED
docker compose exec web sh scripts/smoke_test.sh
```

In (b) and (c), `tests/fixtures/sample_call.wav` (a 20 s business call) should produce one of each moment, with this
summary headline: `we decided to go with the annual plan`, and one action item with owner `Sarah` due `Friday`.

## 6. Open the UI

- Tunnel: `ssh -L 8000:localhost:8000 ubuntu@<instance>`, then browse http://localhost:8000. Microphone live mode works
  this way, because browsers only allow the mic on localhost or HTTPS.
- Direct: http://<public-dns-or-ip>:8000 (needs step 3's extra settings and port 8000 open). Upload works; the
  microphone doesn't over plain HTTP.

## 7. Day-2 commands

```bash
docker compose logs -f web                     # server logs
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build   # redeploy after a code change
docker compose exec web python manage.py process_call /path/in/container.wav   # CLI transcription
docker compose down                            # stop (the data volume with db + model cache is kept)
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `could not select device driver "nvidia"` | NVIDIA Container Toolkit is missing; see step 2 |
| `CUDA driver version is insufficient` | the host driver is older than 570; use the Deep Learning Base AMI or upgrade the driver |
| `healthz` shows `"cuda": false` | you started without `-f docker-compose.gpu.yml` |
| `DisallowedHost` / 400 on the public URL | add the host to `DJANGO_ALLOWED_HOSTS`, then `docker compose up -d` |
| 403 CSRF on the upload form | set `DJANGO_CSRF_TRUSTED_ORIGINS` to the exact origin you browse to |
| model download fails or is rate limited | check outbound HTTPS to huggingface.co; optionally set `HF_TOKEN` in `.env` |
| out of disk during build | grow the root volume to 40 GB+ (`docker system df` shows usage) |

## Earlier design (gfl-copilot, never pushed) and how this one differs

An earlier cloud session designed a copilot under `gfl-copilot/` that never reached a repo. Its handoff was recovered
on 2026-10-12. Kept here so nothing in it is lost:

| | Earlier gfl-copilot | This repo today |
|---|---|---|
| ASR | `nvidia/nemotron-speech-streaming-en-0.6b` **Q8 GGUF** on **NeMo-Speech.cpp v0.2.0** (`nemo-speech serve` on 127.0.0.1:8080, prebuilt CUDA archive, SHA-pinned). Measured ~2.3 ms / 160 ms chunk on a 4090, 2.66% WER; multilingual 3.5 rejected for English (4.58% vs 1.95% WER) | `nvidia/nemotron-3.5-asr-streaming-0.6b` (multilingual) via Transformers, bf16 |
| Decider ("winnow") | **Winnow-12B Q8_0** (`EldanRing/winnow-inference` @ a50c7d0): patched llama.cpp, `/v1/systemone`, CUDA 13.0.2, `winnow-server` on 127.0.0.1:8091, Gemma-4-12B base, 8K ctx, ~143 ms median on a 5070 Ti, run every 1.5 s on partials. File `models/gguf/Winnow-12B-Q8_0.gguf`, 12,669,646,592 bytes, sha256 `b710efc4c0d048ee61eed92c5fef5ce323a4d17e7c51f9f0533cc72ae50818ea` | rule-based `copilot/winnow.py` (regex + lexicon), no LLM |
| Speakers | stereo split (`STEREO_CHANNEL0=agent`), per-leg WebSocket streams | mono first: Nemotron 3 Diarization + role inference; two legs optional |
| Service | FastAPI gateway (Python 3.15, lazy imports, 3.14 fallback) supervising the two native servers; Django UI separate on python:3.14-slim, port 8080 | one Django app (Python 3.12) with UI + JSON API on port 8000 |
| API | `/health`, `/ready`, `POST /v1/calls/analyze`, `POST /v1/calls/{id}/start`, `WS .../legs/{agent|customer}/stream`, `WS .../events`, `DELETE`, optional Bearer auth | `/healthz`, `POST /api/calls/`, live via `POST /api/calls/{id}/partial/`, summary endpoint; no auth |
| Tools | `get_customer_details`, `get_credit_exposure_details`, `get_sales_rep_details` (call start), `validate_tail_number` (≥0.8), `get_fuel_price` (airport mention), `get_order_details`, `get_quote_details` (intent); config in moments.json, questions.json, word_boost.json | `quote_lookup`, `dispute_guidance`, `alert_manager` + decision/action/risk/question/key-moment loggers |
| Hardware | driver ≥ 580, ~35 GB disk, ~13 GB VRAM for Winnow + 1-2 GB ASR (`WINNOW_CONTEXT=4096` on OOM at 16 GB) | driver ≥ 570, ~40 GB disk, ~2 GB VRAM |
| EC2 setup | `./setup.sh --calls-dir /data/gfl-usecases [--cuda-arch 89] [--skip-build]`; CUDA_ARCH from `nvidia-smi --query-gpu=compute_cap` (86 A10G, 89 L4/L40S, 120 50-series / RTX PRO 6000) | `docker compose ... up -d --build` (this file) |

Env keys it used: `CUDA_ARCH, GPU_ID, WINNOW_TARGET=12b-q8, WINNOW_CONTEXT=8192, MODELS_DIR, PYTHON_VERSION=3.15,
CALLS_DIR, CUSTOMERS_CSV, CUSTOMER_ID_COLUMN, STEREO_CHANNEL0=agent, DECIDE_EVERY_S=1.5, GATEWAY_API_KEY, KNOWN_AIRPORTS`.
Known gaps it listed: only stub models tested, lightweight entity extractor (full spec in
`UC4_Entity_Extraction_Agent_Handoff.md`), unauthenticated browser WebSockets, no Five9 VoiceStream bridge.
