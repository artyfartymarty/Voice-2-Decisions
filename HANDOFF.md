# Handoff: run Voice-2-Decisions on EC2 and test it

Instructions for the person or agent setting this up. Follow the steps in order; each lists the command and what
success looks like. Everything here was run on the author's machine (RTX 5070 Ti 16 GB, Docker Desktop) on
2026-10-12/13. The measured results are in section 8.

## What runs

Three containers, started by one `docker compose` command:

| Service | What it is | GPU memory |
|---|---|---|
| `asr` | NVIDIA NeMo-Speech.cpp v0.2.0 (prebuilt CUDA archive, SHA-pinned) serving English Nemotron ASR Q8 and Nemotron 3 Diarization Q8 on :8080 | ~1.5 GB |
| `winnow` | Winnow (EldanRing/winnow-inference @ a50c7d0, built from source): sentiment for the Customer Experience use case on :8091 | E2B ~6 GB / 12B ~13 GB |
| `web` | Django UI + JSON API (Python 3.12, gunicorn) on :8000: speaker roles, winnow rules, the three use-case tools, live streaming diarization | ~1 GB |

Models download on first start into Docker volumes and are checked against pinned SHA-256s.

## 0. What you need

- **EC2 GPU instance.** `g6.xlarge` (L4 24 GB) or `g6e.xlarge` (L40S 48 GB) are the expected choices; `g5.xlarge`
  (A10G 24 GB) also works. With Winnow-12B (`WINNOW_TARGET=q8`), use a 24 GB+ card.
- **AMI:** "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04 or 24.04)". It ships driver 595, Docker and
  the NVIDIA Container Toolkit. **Do not use the ECS GPU AMI**: its driver (550) is too old. The driver must be
  **580 or newer**, because the Winnow image is built on CUDA 13.0.
- **Disk:** at least 80 GB on the root volume. The images are about 25 GB and the models about 7 GB with E2B, or
  20 GB with 12B. The Winnow build also needs room for its compiler image.
- **Outbound HTTPS** to Docker Hub, GitHub, PyPI and huggingface.co, for the build and the first model download.
- **Inbound:** SSH. For the UI, either open TCP 8000 to your IP or use an SSH tunnel (step 6).

## 1. Get the code onto the instance

```bash
git clone https://github.com/artyfartymarty/Voice-2-Decisions.git
cd Voice-2-Decisions
```

If you received the files directly instead, as plain files and not a zip, copy the folder up with `scp -r` or
`rsync` and `cd` into it.

## 2. Check the GPU is visible to Docker, and note its architecture

```bash
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

The driver should be 580 or newer. `compute_cap` gives the architecture: 8.6 is A10G (g5), 8.9 is L4/L40S
(g6/g6e), and 12.0 is Blackwell (g7e, RTX 50-series). If the second command fails, install the NVIDIA Container
Toolkit:

```bash
sudo apt-get install -y nvidia-container-toolkit && sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
```

## 3. Configure

```bash
cp .env.example .env
sed -i "s|^DJANGO_SECRET_KEY=.*|DJANGO_SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_urlsafe(50))')|" .env
```

- `CUDA_ARCH`: the default `86;89;120` builds Winnow for all three GPU families. To build faster, set only
  yours, for example `CUDA_ARCH=89` on g6/g6e (compute_cap 8.9 → 89).
- `WINNOW_TARGET`: keep `e2b` unless you're on a 24 GB+ card and want 12B (`q8`).
- To use the real reference exports instead of the synthetic samples, put `customers.csv` and
  `location_notes.csv` in `./reference`, plus `disputes.csv`, `fleet.csv` and `airport_aliases.csv` if you have
  them. Then uncomment the `./reference:/reference:ro` line in `docker-compose.yml` and set
  `REFERENCE_DIR=/reference`.
- To reach the UI by the instance's address, also set:

```
DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1,<public-dns-or-ip>
DJANGO_CSRF_TRUSTED_ORIGINS=http://<public-dns-or-ip>:8000
```

## 4. Build and start

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
docker compose ps
curl -s "http://localhost:8000/healthz?load=1"
```

The first build compiles Winnow, which takes several minutes. The first start downloads the models. Success:
- `docker compose ps` shows `asr` and `web` as healthy and `winnow` as running.
- `healthz` reports `"backend": "nemo_speech"` with `"ready": true` and `"capabilities": ["asr", "diarization"]`,
  and the diarizer shows `"device": "cuda:0"`.
- `curl -s localhost:8091/health` from inside the network (`docker compose exec web curl -s winnow:8091/health`)
  returns `{"status":"ok"}`.

## 5. Run the tests

```bash
# a) Fast suite: rules, extractor, tail slot, use cases, API, live carry, clients (servers mocked). No GPU.
#    Expect: all passed, a handful skipped (the GPU tests).
docker compose --profile test run --rm test

# b) Real models on the GPU (Transformers path, Nemotron diarization, speech service). Expect: all passed.
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile test run --rm -e RUN_GPU_TESTS=1 test \
  pytest -v -s -rs tests/test_gpu_nemotron.py

# c) End to end against the running stack: upload, live chunks, mono quote call with all use cases.
#    Expect: SMOKE TEST PASSED
docker compose exec web sh scripts/smoke_test.sh

# d) Simulated live call: stream a mono WAV in real time and measure latency. Expect: p95 well under 1 s,
#    and quote_request, dispute_risk and sentiment_shift in first_fired.
docker compose exec web python scripts/stream_wav.py tests/fixtures/quote_call_phone_mono.wav --chunk 2

# e) Sentiment model vs word list on 30 labeled call lines. Expect winnow accuracy around 0.9 or higher.
docker compose exec web python scripts/eval_sentiment.py

# f) Your own calls: speaker accuracy against hand-labeled turns (call.wav + call.truth.json side by side).
docker compose exec web python manage.py eval_speakers /path/in/container/call.wav
```

To stream your own WAVs (16-bit PCM, any rate), copy them in and run `stream_wav.py` on them. The live API is HTTP
chunked: `POST /api/calls/live/`, then `POST /api/calls/<id>/partial/` with `audio` chunks and `final=true` on the last.

## 6. Open the UI

- Tunnel: `ssh -L 8000:localhost:8000 ubuntu@<instance>`, then browse http://localhost:8000. The microphone live
  mode works this way, because browsers only allow the mic on localhost or HTTPS.
- Direct: http://<public-dns-or-ip>:8000. This needs the extra settings from step 3 and port 8000 open. Upload
  works, but the microphone doesn't over plain HTTP.

## 7. Day-2 commands

```bash
docker compose logs -f web asr winnow            # logs
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build   # redeploy after a change
docker compose exec web python manage.py process_call /path/in/container.wav  # CLI: one call -> summary JSON
docker compose down                               # stop (volumes with models and db are kept)
```

## 8. Measured results (RTX 5070 Ti, synthesized test calls)

| What | Result |
|---|---|
| ASR, English Nemotron Q8 on NeMo-Speech.cpp (speech service, warm) | 0% WER on all three test calls; 44 s call in 0.17 s (ASR only), 0.4–0.6 s with diarization |
| ASR, Nemotron 3.5 multilingual via Transformers (fallback `ASR_BACKEND=nemotron`) | 0–2% WER; 44 s call in 0.79 s, plus separate diarization |
| Speaker roles on mono calls (agent / caller) | 8/8 turns; 100% of speech time on the clean call, 99.8% on the phone-quality 8 kHz μ-law call |
| Full pipeline, one 44 s mono call (upload → speakers → transcript → roles → use cases → tools → DB) | 0.45–0.8 s |
| Live, streamed in real time in 2 s chunks | 136 ms p50 / 178 ms p95 per chunk; GP + disputes fire ~0.13 s after the quote is said, the sentiment alert ~0.17 s after the line |
| Sentiment, 30 labeled call lines | Winnow-E2B 93% accuracy, 8/10 negative lines caught, 30 ms per line; word list 60%, 1/10 |
| Entity extraction (customer, location, tail number) | 2.6 ms p95 per transcript update, CPU |
| GPU memory, whole stack with Winnow-E2B | about 8.5 GB on top of the desktop |

These are synthesized TTS calls, which are easier than real phone audio (crosstalk, similar voices, hold music).
Measure real calls with step 5f and `stream_wav.py` before relying on the alerts.

Not yet measured: Winnow-12B against E2B on the same set (the 12B download finished after a GPU pause), and the
14 s cold start of the first live request (the warm-up step still to add).

## Knowledge graph (graphify)

The repo is set up for [graphify](https://github.com/safishamsi/graphify), which maps the code, docs, Django
templates and compose files into a queryable knowledge graph for Claude Code. `CLAUDE.md` tells Claude to query it
before answering codebase questions. The graph itself (`graphify-out/`) is not committed, so build it once per checkout:

```bash
pipx install graphifyy        # or: uv tool install graphifyy / pip install graphifyy
/graphify .                   # in Claude Code: full build (code AST + a semantic pass over docs/templates)
graphify update .             # after code changes: AST-only refresh, no LLM, no cost
graphify hook install         # optional: rebuild automatically after every commit
graphify claude install       # optional: adds per-machine search hooks in .claude/settings.json (keep them local)
graphify query "how does a live chunk reach the sentiment alert?"   # ask the graph
```

On the author's PC the first build produced 633 nodes, 1,442 edges and 33 communities, with the Django UI
(templates, views, URLs) in its own communities. After pulling UI changes, run `/graphify . --update`, so the
changed templates get their semantic pass too.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `could not select device driver "nvidia"` | NVIDIA Container Toolkit missing; see step 2 |
| `CUDA driver version is insufficient` / winnow exits at start | driver older than 580; use the Deep Learning Base AMI, not the ECS GPU AMI |
| winnow fails with "no kernel image is available" | its build didn't include your GPU: set `CUDA_ARCH` from compute_cap and `up -d --build` |
| `healthz` shows `"cuda": false` | started without `-f docker-compose.gpu.yml` |
| sentiment says `lexicon (winnow unavailable)` | winnow not up yet or out of memory: `docker compose logs winnow`; on 16 GB keep `WINNOW_TARGET=e2b` |
| `DisallowedHost` / 400 on the public URL | add the host to `DJANGO_ALLOWED_HOSTS`, then `docker compose up -d` |
| 403 CSRF on the upload form | set `DJANGO_CSRF_TRUSTED_ORIGINS` to the exact origin you browse to |
| model download fails | check outbound HTTPS to huggingface.co and github.com |

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
