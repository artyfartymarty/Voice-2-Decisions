# Voice-2-Decisions

A call copilot. The main input is a **mono call recording** (or a live stream of mono chunks):

```
mono audio ─► Nemotron Diarization ─► Nemotron ASR per turn ─► agent / caller roles ─► winnow ─► tool calls ─► UI / API / alerts
             (who spoke when)          (what they said)          (from what they say)    (what matters)
```

- **Nemotron 3 Diarization** (`nvidia/Nemotron-3-Diarization`, streaming Sortformer) separates the speakers. It
  runs in streaming mode, so a live call keeps the same speaker IDs from chunk to chunk.
- **Nemotron 3.5 ASR** (`nvidia/nemotron-3.5-asr-streaming-0.6b`) transcribes each speaker turn.
- **Roles** (`copilot/roles.py`) decide which diarized speaker is the agent and which is the caller, from what
  they say. The agent greets, looks things up and quotes; the caller names their company, says where they're
  flying and asks for a price.
- **winnow** (`copilot/winnow.py`) picks out decisions, action items, risks, questions and key moments. It also
  runs the three use cases below. Each finding becomes a tool call.

Two separate leg files still work, and each leg is then a speaker. That path is optional; mono is the main one.

For setting it up on EC2, read **[HANDOFF.md](HANDOFF.md)**.

## The three use cases, built to their Pattern

| Category | Pattern | Trigger | Tool | What it returns |
|---|---|---|---|---|
| GP | Static only, Customer and Location required | quote intent, and both customer and location known | `quote_lookup` (flat-file lookup, no model) | addressable and total volume at the quoted location, other locations the customer operates at, margin reduction range (CURRENT to MIN GPPG); **dispatcher alerted** asking for feedback on the quoted price. Outcome: addressable volume checked and **fed back to Yield** when off; **margin flagged high** with a revised mid-range price |
| Avoiding Disputes | Single key word, static data search | a location in play plus a keyword (quote, fee, price, callout...) | `dispute_guidance` (notes + dispute list, no model) | fires when the location or the customer is dispute-prone (with rank); prompt to restate fees, and written quote wording. Outcome: get ahead of a credit or rebill |
| Customer Experience | Sentiment and orchestration | a speaker's sentiment turns negative | `alert_manager` (**sentiment model: Winnow** `/v1/systemone`, lexicon fallback) | prompt to the agent on the call, real-time alert to the designated managers / leads (`MANAGER_WEBHOOKS`) |

Each result carries a card with the sheet's columns: **Who, Want to see, Outcome, Category** (plus Pattern). The
summary returns them as `use_cases`, and the call page shows them as a table.

Reference data comes from flat files in `REFERENCE_DIR`, with the same columns as the work exports. `disputes.csv`, `fleet.csv` and `airport_aliases.csv` are optional and synthetic here:
- `customers.csv`: CUSTOMER_NAME, CUSTOMER_NUMBER, ICAO, P12M_TOTAL_WFS_VOL, P12M_ADDRESSABLE_VOL, MIN_GPPG, CURRENT_GPPG
- `location_notes.csv`: ICAO, IATA, Country, Region, Supplier, FBO_NUMBER, FBO, Private Notes, Public Notes (messy HTML is fine)

`fixtures/reference/` holds **synthetic** samples in those shapes; none of it is real customer data. Alerts are
logged, and also POSTed to `NOTIFY_WEBHOOK_URL` when that is set, for example a Slack or Teams webhook.

## Tool arguments without an LLM

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

## Quick start (Docker, GPU)

```bash
cp .env.example .env            # then set DJANGO_SECRET_KEY
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
curl "http://localhost:8000/healthz?load=1"     # downloads both models on first run (~3 GB), then reports the GPU
```

Open http://localhost:8000 to upload a recording or record live from the microphone.

## Tests

```bash
docker compose --profile test run --rm test                      # fast suite, no GPU or model needed
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile test run --rm -e RUN_GPU_TESTS=1 test pytest -v -s -rs tests/test_gpu_nemotron.py
docker compose exec web sh scripts/smoke_test.sh                 # end to end against the running server
docker compose exec web python manage.py eval_speakers /path/call.wav   # needs call.truth.json next to it
```

## API

| Method | Path | What it does |
|---|---|---|
| GET | `/healthz` (`?load=1` loads the models) | ASR + diarizer status, GPU, registered tools |
| POST | `/api/calls/` multipart `audio` (mono), optional `audio_b` + `speaker_a`/`speaker_b` for two legs, `title` | process a recording → transcript with speakers, moments, tool results (201) |
| GET | `/api/calls/<id>/` | full call: segments with speaker and role, moments, context |
| GET | `/api/calls/<id>/summary/` | headline, counts, decisions, action items, risks, questions, quotes, dispute warnings, sentiment alerts, speaker roles |
| POST | `/api/calls/live/` JSON `{"title"}` | start a live call |
| POST | `/api/calls/<id>/partial/` multipart `audio` chunk (+ `final`), or JSON `{"text", "speaker", "final"}` | feed a live call; returns new moments and the current speaker roles |

CLI: `python manage.py process_call recording.wav [--full]` prints the summary as JSON.

## Layout

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

## Accuracy and limits

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

## Running without Docker

Python 3.12, ffmpeg on PATH, then:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt -r requirements-dev.txt
python manage.py migrate && python manage.py runserver
```
