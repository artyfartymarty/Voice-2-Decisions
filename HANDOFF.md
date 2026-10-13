# Handoff: Voice-2-Decisions on EC2

For the person or agent setting this up. **Setup is in the [README](README.md):** clone, run `./scripts/setup.sh`,
tunnel to :8080. This file holds what's behind it: how the pieces fit, what was measured, how to troubleshoot, and
reference material.

## What runs

| Service | What it is | GPU memory |
|---|---|---|
| `asr` | NVIDIA NeMo-Speech.cpp v0.2.0 (prebuilt CUDA archive from GitHub, SHA-pinned; kernels for sm_75/86/89/120) serving English Nemotron ASR Q8 + Nemotron 3 Diarization Q8 on :8080 inside the network | ~1.5 GB |
| `winnow` | Winnow-12B Q8 on winnow-server (EldanRing/winnow-inference @ a50c7d0, built from GitHub source for `CUDA_ARCH`) at 4K context: sentiment for Customer Experience, on :8091 | ~12.5 GB |
| `web` | Django UI + JSON API (Python 3.12, gunicorn, host port 8080): speaker roles, winnow rules, the use-case tools, live streaming diarization (Nemotron-3-Diarization from `./models/diar-hf`, offline) | ~1 GB |

All model files live in `./models` on the host (see README, "Use your own model files"). Nothing downloads at
container start.

## Choosing an instance

- **`g5.2xlarge` (A10G 24 GB): the target.** Whole stack is about 15 GB of VRAM with no display attached, so it
  fits with headroom.
- `g6.xlarge`/`g6e.xlarge` (L4 24 GB / L40S 48 GB) also work; `setup.sh` sets `CUDA_ARCH=89`.
- `g4dn.xlarge` (T4 16 GB) runs every component (sm_75 kernels everywhere, float16 instead of bfloat16), but 12B
  plus speech is tight on 16 GB. Expect slower sentiment and keep nothing else on the GPU.
- **AMI:** Deep Learning Base OSS Nvidia Driver GPU AMI (driver 595). Not the ECS GPU AMI: its driver 550 is too
  old (Winnow is built on CUDA 13, which needs 580+).
- **Disk:** 100 GB. Images are ~25 GB, models ~14 GB, plus the Winnow build cache.

## What `setup.sh` does

1. Checks `nvidia-smi`: driver 580+, GPU memory, and `compute_cap`, which becomes `CUDA_ARCH` (8.6 → 86).
2. Checks Docker can see the GPU (NVIDIA Container Toolkit).
3. Writes `.env` from `.env.example` with a fresh secret key (an existing `.env` is kept; only `CUDA_ARCH` updates).
4. `scripts/fetch_models.sh`: downloads missing model files from `models.lock`, verifying size + SHA-256. With
   `MODELS_SOURCE=local` it only verifies files you copied in.
5. `docker compose build`, then `docker compose up -d`. The first Winnow build compiles CUDA kernels for your GPU
   (several minutes).
6. `check.py --stack`: settings, migrations, fast tests, then `doctor --golden` on the real models. Must end in
   `check: all passed`.

## Measured results (RTX 5070 Ti, synthesized test calls)

| What | Result |
|---|---|
| ASR, English Nemotron Q8 on NeMo-Speech.cpp (speech service, warm) | 0% WER on all three test calls; 44 s call in 0.17 s (ASR only), 0.4–0.6 s with diarization |
| ASR, Nemotron 3.5 multilingual via Transformers (fallback `ASR_BACKEND=nemotron`) | 0–2% WER; 44 s call in 0.79 s, plus separate diarization |
| Speaker roles on mono calls (agent / caller) | 8/8 turns; 100% of speech time on the clean call, 99.8% on the phone-quality 8 kHz μ-law call |
| Full pipeline, one 44 s mono call (upload → speakers → transcript → roles → use cases → tools → DB) | 0.45–0.8 s |
| Live, streamed in real time in 2 s chunks | 136 ms p50 / 178 ms p95 per chunk; GP + disputes fire ~0.13 s after the quote is said, the sentiment alert ~0.17 s after the line |
| Sentiment, 30 labeled call lines | Winnow-E2B 93% accuracy, 8/10 negative lines, 27–30 ms per line · Winnow-12B (4K context) 97%, 10/10 negative lines, 73 ms · word list 60%, 1/10 |
| Winnow-12B context | 8K context needs 17.1 GB, so on a 16 GB card Windows spills to system RAM: 585 ms/line. 4K fits: 70 ms (prefill 346 → 35 ms) |
| Live, 1 s chunks (after the cadence-aware flush) | 66–73 ms p50, all use cases fire, words intact |
| Cold start | first live request no longer waits: models warm up at server start (~17 s, in the background) |
| `scripts/check.py --stack` | settings, migrations, tests, doctor all PASS; golden quote call fires quote_lookup, dispute_guidance, alert_manager |
| Entity extraction (customer, location, tail number) | 2.6 ms p95 per transcript update, CPU |
| GPU memory, whole stack with Winnow-12B (4K) | ~13.5 GB on top of the desktop. On the 16 GB dev card that also drives monitors it left too little for the displays and slowed to ~0.6 s per sentiment line; the 24 GB g5 has headroom |

These are synthesized TTS calls, which are easier than real phone audio (crosstalk, similar voices, hold music).
Measure real calls with step 5f and `stream_wav.py` before relying on the alerts.

Not yet measured: the whole stack with Winnow-12B on a 24 GB card. Run `check.py --stack`, `eval_sentiment.py` and
`stream_wav.py` on the g5 and compare with the table.

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
| `setup.sh`: driver older than 580 | use the Deep Learning Base OSS Nvidia Driver GPU AMI (driver 595), not the ECS GPU AMI |
| `Docker can't see the GPU` | `sudo apt-get install -y nvidia-container-toolkit && sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker` |
| `MISSING` / `BAD` from fetch_models.sh | copy the file to the path shown (README, "Use your own model files"), or unset `MODELS_SOURCE=local` to download it |
| `asr` exits with "missing /models/..." | models aren't in `./models`; run `scripts/fetch_models.sh` |
| winnow logs "no kernel image is available" | it was built for another GPU: fix `CUDA_ARCH` in `.env` (from compute_cap) and `docker compose build winnow` |
| sentiment says `lexicon (winnow unavailable)` | winnow still loading or out of memory: `docker compose logs winnow`; keep `WINNOW_CONTEXT=4096` |
| everything slow, GPU memory near full | something else is on the GPU (or a display on a 16 GB card): 12B + speech needs ~15 GB |
| `DisallowedHost` / CSRF 403 on a public URL | use the SSH tunnel, or add the host to `DJANGO_ALLOWED_HOSTS` and `DJANGO_CSRF_TRUSTED_ORIGINS` |

## Reference (moved from the README)

### The three use cases, built to their Pattern

| Category | Pattern | Trigger | Tool | What it returns |
|---|---|---|---|---|
| GP | Static only, Customer and Location required | quote intent, and both customer and location known | `quote_lookup` (flat-file lookup, no model) | addressable and total volume at the quoted location, other locations the customer operates at, margin reduction range (CURRENT to MIN GPPG); **dispatcher alerted** asking for feedback on the quoted price. Outcome: addressable volume checked and **fed back to Yield** when off; **margin flagged high** with a revised mid-range price |
| Avoiding Disputes | Single key word, static data search | a location in play plus a keyword (quote, fee, price, callout...) | `dispute_guidance` (notes + dispute list, no model) | fires when the location or the customer is dispute-prone (with rank); prompt to restate fees, and written quote wording. Outcome: get ahead of a credit or rebill |
| Customer Experience | Sentiment and orchestration | a speaker's sentiment turns negative | `alert_manager` (**sentiment model: Winnow** `/v1/systemone`, lexicon fallback) | prompt to the agent on the call, real-time alert to the designated managers / leads (`MANAGER_WEBHOOKS`) |

Each result carries a card with the sheet's columns: **Who** (always Customer), **Want to see, Outcome, Category**, plus Pattern and who was alerted. The
summary returns them as `use_cases`, and the call page shows them as a table.

Reference data comes from flat files in `REFERENCE_DIR`, with the same columns as the work exports. `disputes.csv`, `fleet.csv` and `airport_aliases.csv` are optional and synthetic here:
- `customers.csv`: CUSTOMER_NAME, CUSTOMER_NUMBER, ICAO, P12M_TOTAL_WFS_VOL, P12M_ADDRESSABLE_VOL, MIN_GPPG, CURRENT_GPPG
- `location_notes.csv`: ICAO, IATA, Country, Region, Supplier, FBO_NUMBER, FBO, Private Notes, Public Notes (messy HTML is fine)

`fixtures/reference/` holds **synthetic** samples in those shapes; none of it is real customer data. Alerts are
logged, and also POSTed to `NOTIFY_WEBHOOK_URL` when that is set, for example a Slack or Teams webhook.

### Tool arguments without an LLM

The decider only picks *which* tool fires. The arguments come from `copilot/extract.py` and `copilot/slots.py`,
which are deterministic, about 2.5 ms p95 per update on CPU, and can only return values from the reference data or
a strict grammar:

- **Customers:** account numbers (written or spoken digit by digit) or names, matched fuzzily and phonetically
  against customers.csv ("North Wind aviation" → NORTHWIND AVIATION, LLC). They can also be pre-filled at call
  start with `customer_id` (from the phone system's ANI lookup).
- **Locations:** ICAO and IATA codes, also spelled ("K T E B") or in the phonetic alphabet. FBO names come from
  the notes; airport and city names come from `airport_aliases.csv`, fuzzy ("Teeterborough" → KTEB).
- **Tail numbers:** registration grammar per country, with phonetic alphabet and spoken digits ("November five one
  two bravo tango" → N512BT). A per-call slot handles cut-offs, re-asks ("sorry, what was the tail again?"),
  spelling spread over several breaths, and corrections ("no, B T, not B D"). `validate_tail_number` fires only
  for a complete, confident value, or one found in the customer's fleet (`fleet.csv`). Anything less becomes an
  **agent prompt** ("Tail number incomplete, ask the caller to repeat it. On file: N512BT").
- **Word boost:** `GET /api/boost-words/` lists the same names and codes, to feed the ASR's word-boost list.

### API

| Method | Path | What it does |
|---|---|---|
| GET | `/healthz` (`?load=1` loads the models) | ASR + diarizer status, GPU, registered tools |
| POST | `/api/calls/` multipart `audio` (mono), optional `audio_b` + `speaker_a`/`speaker_b` for two legs, `title` | process a recording → transcript with speakers, moments, tool results (201) |
| GET | `/api/calls/<id>/` | full call: segments with speaker and role, moments, context |
| GET | `/api/calls/<id>/summary/` | headline, counts, decisions, action items, risks, questions, quotes, dispute warnings, sentiment alerts, speaker roles |
| POST | `/api/calls/live/` JSON `{"title"}` | start a live call |
| POST | `/api/calls/<id>/partial/` multipart `audio` chunk (+ `final`), or JSON `{"text", "speaker", "final"}` | feed a live call; returns new moments and the current speaker roles |
| GET | `/calls/<id>/` | the call page: who's calling, tail-number hint, prompt cards per use case, live transcript, summary when the call ends |
| GET | `/calls/<id>/panels/` (`?v=<version>`) | the call page's panels as HTML for the live script; 204 when nothing changed |
| GET | `/alerts/` (`?live=1`) | manager alert view: sentiment and quote alerts across calls, newest first |

While a call is live the call page refreshes its panels every `LIVE_REFRESH_MS` (1.5 s). Set `GATEWAY_WS_URL` to a
WebSocket that announces call updates and the page also refreshes on each message.

CLI: `python manage.py process_call recording.wav [--full]` prints the summary as JSON.

### Layout

| Path | Role |
|---|---|
| `copilot/diarize.py` | Nemotron Diarization in streaming mode; one session per live call |
| `copilot/asr.py` | Nemotron ASR (utterance-segmented), `FakeASR` for tests |
| `copilot/roles.py` | names diarized speakers agent / caller from their words |
| `copilot/winnow.py` | the decision engine: clause splitting, rules, per-speaker sentiment, call context (customer, location, quote intent) |
| `copilot/reference.py` | loads the two flat files, cleans the notes HTML, finds customers and locations in speech |
| `copilot/usecases.py` | `quote_lookup`, `dispute_guidance`, `alert_manager`, and `notify()` |
| `copilot/tools.py` | tool registry; calls to tools that don't exist yet stay `pending` and are replayed later |
| `copilot/pipeline.py` | glue for recordings and live calls |
| `copilot/evaluation.py` | role-attribution accuracy against hand-labeled turns |
| `tests/fixtures/` | synthesized calls: `sample_call.wav`; the quote call as two legs, as clean mono, and as phone-quality mono, each with `.truth.json` |

### Accuracy and limits

- **Speaker attribution** was measured on the synthesized quote call (two TTS voices) with the real models; see
  HANDOFF.md for the numbers. Real calls are harder: similar voices, crosstalk, very short replies like
  "yeah" or "ok", and hold music. Expect lower numbers there, and measure them with `eval_speakers` on a few
  hand-labeled calls before trusting the alerts.
- **Roles** come from content. A call with no greeting and no quote request gives weak evidence, and the summary
  then reports `speaker_roles_confident: false`. Labels fall back to "whoever spoke first is the agent".
- **winnow** is rule based (regex and lexicon), so it's fast, deterministic and testable, but it only catches
  phrasings its rules cover. Its rules are in `copilot/winnow.py` and `copilot/roles.py`.
- Live mode transcribes each chunk separately, so a word cut at a chunk boundary can be misheard.
- Microphone capture in the browser needs `localhost` or HTTPS.
- Requests run synchronously in one gunicorn worker with 4 threads, and the models are shared under locks. Live
  diarization sessions live in that worker's memory, so keep `--workers 1`.

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
