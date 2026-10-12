# Voice-2-Decisions

Turns call audio into decisions. Audio goes through **Nemotron ASR** (`nvidia/nemotron-3.5-asr-streaming-0.6b`).
The transcript then goes through **winnow**, which picks out decisions, action items (with owner and due date),
risks, open questions and key moments (prices, budgets, deadlines). Each moment becomes a **tool call**. A
**Django** web UI and JSON API sit on top.

```
audio (1 file, or 2 call legs) ──► mix to mono 16 kHz ──► Nemotron ASR ──► winnow ──► tool calls ──► SQLite + UI/API
live mic / chunks ───────────────► partial transcripts ──┘   (decides as soon as each sentence ends)
```

For setting it up on another machine (EC2), read **[HANDOFF.md](HANDOFF.md)**.

## Quick start (Docker, GPU)

```bash
cp .env.example .env            # then set DJANGO_SECRET_KEY
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
curl "http://localhost:8000/healthz?load=1"     # downloads the model on first run (~2.5 GB), then reports the GPU
```

Open http://localhost:8000. You can upload a recording (optionally a second leg) or record live from the microphone.

## Tests

```bash
docker compose --profile test run --rm test                      # 35 fast tests, no GPU or model needed
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile test run --rm -e RUN_GPU_TESTS=1 test pytest -v -rs tests/test_gpu_nemotron.py
docker compose exec web sh scripts/smoke_test.sh                 # end to end against the running server
```

## API

| Method | Path | What it does |
|---|---|---|
| GET | `/healthz` (`?load=1` to load the model) | backend, GPU, registered tools |
| POST | `/api/calls/` multipart `audio`, optional `audio_b`, `title` | transcribe a recording, return transcript + moments (201) |
| GET | `/api/calls/<id>/` | full call: segments, transcript, moments with tool call status |
| GET | `/api/calls/<id>/summary/` | headline, counts, decisions, action items, risks, questions, key moments |
| POST | `/api/calls/live/` JSON `{"title"}` | start a live call |
| POST | `/api/calls/<id>/partial/` JSON `{"text","final"}` or multipart `audio` chunk + `final` | feed a live call; returns new moments |

CLI: `python manage.py process_call recording.wav [second_leg.wav] [--full]` prints the summary as JSON.

## Layout

| Path | Role |
|---|---|
| `copilot/audio.py` | decode any format (libsndfile, ffmpeg fallback), mix two legs to mono, cut long audio at quiet points |
| `copilot/asr.py` | `NemotronASR` (Transformers, bf16 on GPU) and `FakeASR` (tests / CPU-only) |
| `copilot/winnow.py` | the decision engine: sentence + clause splitting, rules per moment kind, incremental `Winnower` for partials, summaries |
| `copilot/tools.py` | tool registry. Add integrations with `@tool("create_action_item")`; calls to unregistered tools are kept as `pending` and `pipeline.replay_pending()` runs them later |
| `copilot/pipeline.py` | glue: recording or partial → ASR → winnow → tools → database |
| `copilot/views.py`, `templates/` | web UI and JSON API |
| `tests/` | unit + API tests (FakeASR), GPU tests (real model), `fixtures/sample_call.wav` (20 s synthesized business call) |

## Notes and limits

- winnow is rule based, so it's fast, deterministic and testable. Its rules are in `copilot/winnow.py`. It
  doesn't know who is speaking: "I'll send it" gets owner "I". Speaker diarization would be the next step.
- Live mode transcribes each audio chunk on its own, so a word cut at a chunk boundary can be misheard. Moving to
  the model's cache-aware streaming API would fix this. Text partials, for example from another ASR, have no such issue.
- Microphone capture in the browser needs `localhost` or HTTPS. Over plain HTTP on EC2, use upload or the API.
- Requests run synchronously, with one gunicorn worker and 4 threads, and the model is shared under a lock. A
  20 s call takes about 0.7 s on an RTX 5070 Ti; 3.4 min takes a few seconds. The model uses about 1.6 GB of VRAM.

## Running without Docker

Python 3.12, ffmpeg on PATH, then:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt -r requirements-dev.txt
python manage.py migrate && python manage.py runserver
```
