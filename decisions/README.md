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
no GPU). Everything the build downloads comes from GitHub (the Winnow server is built from
`github.com/EldanRing/winnow-inference`). The 12.7 GB model file is the one thing you bring: copy it onto the instance
(below), and nothing touches Hugging Face.

## Run it on AWS

**Instance:** `g5.2xlarge` (A10G 24 GB) · **AMI:** *Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)* ·
**Disk:** 100 GB · **Security group:** SSH only.

```bash
git clone https://github.com/artyfartymarty/Voice-2-Decisions.git
cd Voice-2-Decisions/decisions
mkdir -p models/winnow/gguf          # then copy the model in: see "Get the model onto the instance"
./setup.sh
```

`setup.sh` checks the GPU and Docker, writes `.env`, checks the model file (12.7 GB, size and SHA-256),
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

## Get the model onto the instance

The model must end up at exactly this path, with exactly this size and hash:

```
decisions/models/winnow/gguf/Winnow-12B-Q8_0.gguf   12669646592 bytes
sha256 b710efc4c0d048ee61eed92c5fef5ce323a4d17e7c51f9f0533cc72ae50818ea
```

**Copy it from your machine (the default, `MODELS_SOURCE=local`).** Nothing is downloaded; `setup.sh` only checks the
file and stops with `MISSING` or `BAD` (and the expected size and hash) if it's wrong.

```powershell
# from Windows (PowerShell), wherever your copy of the file is
scp C:\path\to\Winnow-12B-Q8_0.gguf ubuntu@<instance-public-dns>:Voice-2-Decisions/decisions/models/winnow/gguf/
```

```bash
# or via an S3 bucket: upload once from your laptop, then on the instance
aws s3 cp s3://<bucket>/Winnow-12B-Q8_0.gguf models/winnow/gguf/
```

Check it any time with `scripts/fetch_model.sh`.

**Or let `setup.sh` download it** by changing `.env`:

| `.env` | Where the file comes from |
|---|---|
| `MODELS_SOURCE=local` | you copy it in (default) |
| `MODELS_SOURCE=url` + `MODEL_URL=https://...` | one download from any direct link: your S3 bucket, a presigned URL, your own server |
| `MODELS_SOURCE=github` | the parts of a GitHub release (`MODEL_REPO`, `MODEL_TAG`). Release files must be under 2 GB, so someone has to upload it once as 7 parts with `python decisions/scripts/publish_model.py Winnow-12B-Q8_0.gguf` (needs `gh auth login`) |

Every download resumes if interrupted and is checked against the SHA-256 above. Winnow-12B is Apache-2.0 (a Gemma 4
fine-tune by EldanRing), so copying or re-hosting it with the license is allowed.

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
- **Tail numbers:** set `TAIL_NORMALIZER_URL` (and `TAIL_NORMALIZER_MODEL` for an OpenAI-compatible endpoint) to the
  model that writes tail numbers. When a caller line is about a tail (or answers the agent asking for one), the same
  Winnow request asks "did the caller just say a complete tail number?"; on yes, that line goes to the model, its
  answer must be a valid registration and, if the customer has a fleet on file, one of their tails, and the agent is
  prompted to read it back. Without `TAIL_NORMALIZER_URL`, or with `TAIL_MODE=parser`, the built-in parser reads tails
  instead (see Notes). See `normalizer.py`.
- **How sure Winnow must be:** `DECIDE_MIN_P` (default 0.5).
- **Fuzzy names:** customer and location are matched against the reference lists first. When a match came from a heard
  name rather than an exact code or account number, the same Winnow request also asks "which customer / airport did
  they mean?" over the closest few names plus "none of these", and its pick settles the variable (functions re-run if
  it changed). The agent is also prompted to read the name back. Each decision prints its round-trip time (`ms`).

| File | What it does |
|---|---|
| `events.py` | Nova Sonic / plain events -> (speaker, text, final) |
| `engine.py` | one call: variables, auto-calling, decisions |
| `functions.py` | the functions and the variables they need |
| `normalizer.py` | the external model that writes tail numbers |
| `decider.py` | Winnow-12B `choice` question on `/v1/systemone`, keyword fallback |
| `server.py`, `console.html` | the HTTP API and the console |
| `replay.py` | feed a script, Nova events or typed lines |
| `scripts/fetch_model.sh` | checks the model file, or downloads it (`MODELS_SOURCE`) |
| `scripts/publish_model.py` | optional: put the model on a GitHub release in parts |

## Notes: omitted for now

- **Tail numbers go to the external model, not the parser.** The deterministic tail parser (`copilot/slots.py`:
  phonetic alphabet, aviation digits, "one twenty three", a dropped "November", tails over several breaths) is not the
  main path. It is used only with `TAIL_MODE=parser` or when no `TAIL_NORMALIZER_URL` is set.
- **No special handling of mid-tail corrections** ("forget the November, I meant tango", "scratch that") on the main
  path: the model sees the caller's line and the few before it and is trusted to resolve them; the format and fleet
  checks and the agent's read-back are the safety net. (The parser handles them when it's in use, and its tests stay.)
