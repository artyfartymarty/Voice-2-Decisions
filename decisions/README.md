# Text to Decisions

Nova Sonic turns the call's speech into text. This service takes that text and does the rest:

1. **Variables.** Each final line is parsed for the customer (account number or company name), the location (ICAO,
   IATA, FBO or airport name) and the aircraft tail number. Only values that exist in the reference data, or a valid
   registration, can fill a variable.
2. **Functions fire on their own.** A function is called the moment every variable it needs is filled, and again if
   one of its inputs changes. Results wait in the call.
3. **The decision model picks the answer.** After each caller line, **Winnow-12B** decides what the caller is asking
   for and surfaces that function's result. If the caller asked before the variables were filled, the decision tells
   the agent what to ask for, and the result comes up as soon as the last variable arrives.

```
"Northwind, account 204511"            -> customer_number=204511          -> customer_profile()
"flying into Van Nuys, price per gallon?" -> icao=KVNY                    -> fuel_quote(), dispute_check(), location_info()
                                          Winnow-12B: caller wants fuel_quote (p=0.93) -> "Quote Northwind at KVNY: ..."
```

| Function | Needs | Surfaced when the caller... |
|---|---|---|
| `fuel_quote` | customer_number, icao | wants a price, quote or rate |
| `dispute_check` | icao (+ customer) | questions a fee or charge, asks what fees apply, threatens a dispute |
| `location_info` | icao | asks about hours, notice or how to arrange fuel |
| `tail_check` | tail_number (+ customer) | gives a tail number or asks whether it's on the account |
| `customer_profile` | customer_number | asks what's on file for their account |
| `escalate` | nothing; never auto-called | is angry, threatens to leave, or asks for a manager |

Two containers: **winnow** (Winnow-12B on the GPU, ~14 GB VRAM) and **api** (Python standard library, 180 MB image,
no GPU). The 12B model is downloaded from this repo's **GitHub release**, never from Hugging Face.

## Run it on AWS

**Instance:** `g5.2xlarge` (A10G 24 GB) · **AMI:** *Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)* ·
**Disk:** 100 GB · **Security group:** SSH only.

```bash
git clone https://github.com/artyfartymarty/Voice-2-Decisions.git
cd Voice-2-Decisions/decisions
./setup.sh
```

`setup.sh` checks the GPU and Docker, writes `.env`, downloads Winnow-12B from GitHub (12.7 GB, checked by SHA-256),
builds both images (Winnow compiles for your GPU, which takes a while the first time), starts them, and replays a demo
call through the API. Each decision in that output should say `[winnow, p=...]`.

From your laptop:

```bash
ssh -L 8000:localhost:8000 ubuntu@<instance-public-dns>
# open http://localhost:8000
```

The console lets you type caller and agent lines, or replay the three demo calls, and shows the variables, which
functions have fired, and what the decision model surfaced.

After a reboot: `./setup.sh --skip-build`, or just `docker compose up -d`.

## Put the model on GitHub (once)

GitHub release files must be under 2 GB, so the 12.7 GB GGUF goes up as 7 parts. On any machine that already has
`Winnow-12B-Q8_0.gguf` and the GitHub CLI logged in:

```bash
gh auth login
python decisions/scripts/publish_model.py path/to/Winnow-12B-Q8_0.gguf
```

It checks the file's SHA-256, creates the release `models-v1`, and uploads the parts one at a time (needs ~2 GB free
disk; re-running skips parts that are already up). `scripts/fetch_model.sh` downloads them from
`https://github.com/artyfartymarty/Voice-2-Decisions/releases/download/models-v1/`, joins them and checks the result.
To use another repo or tag, set `MODEL_REPO` / `MODEL_TAG` in `.env`. Already copied the file to the instance? Set
`MODELS_SOURCE=local` and it is only verified.

Winnow-12B is Apache-2.0 (a Gemma 4 fine-tune by EldanRing), so redistributing it with the license is allowed; the
release notes carry it. The Winnow server itself is built from `github.com/EldanRing/winnow-inference` at a pinned
commit.

## Send it text

`POST /calls/<call-id>/events` with one event or a list. Any of these shapes works:

```jsonc
// Nova Sonic, raw stream events (the stage comes from contentStart.additionalModelFields)
{"event": {"contentStart": {"contentId": "c1", "type": "TEXT", "role": "USER",
                            "additionalModelFields": "{\"generationStage\":\"FINAL\"}"}}}
{"event": {"textOutput": {"contentId": "c1", "role": "USER", "content": "I'd like a fuel quote at Teterboro"}}}

// Nova Sonic textOutput, flattened
{"role": "USER", "content": "I'd like a fuel quote at Teterboro", "generationStage": "FINAL"}

// plain
{"speaker": "caller", "text": "I'd like a fuel quote at Teterboro"}
```

`USER` is the caller and `ASSISTANT` the agent (change with `ROLE_MAP`). Only `FINAL` text fills variables;
`SPECULATIVE` text is shown as a preview and otherwise ignored.

```bash
curl -s localhost:8000/calls/abc/events -H 'Content-Type: application/json' \
  -d '{"role":"USER","content":"Northwind, account 204511. Price per gallon at Teterboro?","generationStage":"FINAL"}'
```

The response lists, for each line, the variables that changed, the functions that fired with their results, any agent
prompts (such as an incomplete tail number), and the decision:

```json
{"decision": {"function": "fuel_quote", "ready": true, "by": "winnow", "p": 0.93,
              "say": "Quote NORTHWIND AVIATION, LLC at KTEB: current margin 0.3125 GPPG, room to come down 0.2275 to the 0.0850 floor. ...",
              "result": {"current_gppg": 0.3125, "floor_gppg": 0.085, "...": "..."},
              "because": "Northwind, account 204511. Price per gallon at Teterboro?"},
 "variables": {"customer_number": "204511", "icao": "KTEB"}}
```

Other endpoints: `GET /calls/<id>` (full call state), `DELETE /calls/<id>`, `GET /functions`, `GET /healthz`.
Set `DECISION_WEBHOOK_URL` to have every new decision POSTed to your app as well.

## Test without a GPU

```bash
python -m unittest decisions.test_decisions -v                       # from the repo root, Python 3.10+
python -m decisions.replay tests/fixtures/demo_frustrated_caller.script.json
python -m decisions.server                                           # console on :8000, keyword decider
echo "caller: Northwind at Van Nuys, what's your price?" | python -m decisions.replay -
```

With no `WINNOW_URL` the decider is a keyword fallback, so the whole flow runs on a laptop. The same fallback covers
the API while Winnow is loading or if it stops answering.

## Change it

- **Add a function:** append a `Function` in `functions.py` with its `needs`, a `description` written as "the caller
  ..." (the decision model reads it), and the code that runs. It fires and becomes an option for Winnow automatically.
- **Real data:** put `customers.csv`, `location_notes.csv` (and optionally `disputes.csv`, `fleet.csv`,
  `airport_aliases.csv`) in `decisions/reference`, uncomment the volume in `docker-compose.yml`, set
  `REFERENCE_DIR=/reference`.
- **How sure Winnow must be:** `DECIDE_MIN_P` (default 0.5).

| File | What it does |
|---|---|
| `events.py` | Nova Sonic / plain events -> (speaker, text, final) |
| `engine.py` | one call: variables, auto-calling, decisions |
| `functions.py` | the functions and the variables they need |
| `decider.py` | Winnow-12B `choice` question on `/v1/systemone`, keyword fallback |
| `server.py`, `console.html` | the HTTP API and the console |
| `replay.py` | feed a script, Nova events or typed lines |
| `scripts/fetch_model.sh`, `scripts/publish_model.py` | the model on GitHub releases |
