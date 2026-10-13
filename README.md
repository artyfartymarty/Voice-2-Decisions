# Voice-2-Decisions

Nova Sonic turns the call's speech into text. This service takes that text and does the rest:

1. **Variables.** Each final line is checked for the customer, the location and the aircraft tail number. Winnow-12B
   decides whether the caller just said a tail number or their company name; if so, GPT 6 Luna pulls it out. Account
   numbers and locations (ICAO, IATA, FBO or airport name) are matched against the reference lists. Only values that
   exist in the reference data, or a valid registration, can fill a variable, and the agent reads names and tails back.
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
cd Voice-2-Decisions
mkdir -p models/winnow/gguf          # then copy the model in: see "Get the model onto the instance"
./setup.sh
```

`setup.sh` checks the GPU and Docker, writes `.env`, checks the model file (12.7 GB, size and SHA-256),
builds both images (Winnow compiles for your GPU, which takes a while the first time), starts them, and replays a demo
call through the API. Each decision in that output should say `[winnow, p=...]`.

Feed it text from the instance (or through `ssh -L 8000:localhost:8000 ubuntu@<instance>` from your laptop):

```bash
docker compose exec api python -m decisions.replay --url http://localhost:8000 demos/demo_frustrated_caller.json
echo "caller: Northwind at Van Nuys, what's your price?" | docker compose exec -T api python -m decisions.replay --url http://localhost:8000 -
```

Each line prints the variables that filled, the functions that fired, agent prompts, and the decision.

After a reboot: `./setup.sh --skip-build`, or just `docker compose up -d`.

## Get the model onto the instance

The model must end up at exactly this path, with exactly this size and hash:

```
models/winnow/gguf/Winnow-12B-Q8_0.gguf   12669646592 bytes
sha256 b710efc4c0d048ee61eed92c5fef5ce323a4d17e7c51f9f0533cc72ae50818ea
```

**Copy it from your machine (the default, `MODELS_SOURCE=local`).** Nothing is downloaded; `setup.sh` only checks the
file and stops with `MISSING` or `BAD` (and the expected size and hash) if it's wrong.

```powershell
# from Windows (PowerShell), wherever your copy of the file is
scp C:\path\to\Winnow-12B-Q8_0.gguf ubuntu@<instance-public-dns>:Voice-2-Decisions/models/winnow/gguf/
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
| `MODELS_SOURCE=github` | the parts of a GitHub release (`MODEL_REPO`, `MODEL_TAG`). Release files must be under 2 GB, so someone has to upload it once as 7 parts with `python scripts/publish_model.py Winnow-12B-Q8_0.gguf` (needs `gh auth login`) |

Every download resumes if interrupted and is checked against the SHA-256 above. Winnow-12B is Apache-2.0 (a Gemma 4
fine-tune by EldanRing), so copying or re-hosting it with the license is allowed.

## Send it text

The full request and response contract, for wiring up the UI, is in [API.md](API.md). In short:

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
python -m decisions.replay demos/demo_frustrated_caller.json
python -m decisions.server                                           # API on :8000, keyword decider
echo "caller: Northwind at Van Nuys, what's your price?" | python -m decisions.replay -
```

With no `WINNOW_URL` the decider is a keyword fallback, so the whole flow runs on a laptop. The same fallback covers
the API while Winnow is loading or if it stops answering.

## Change it

- **Add a function:** append a `Function` in `decisions/functions.py` with its `needs`, a `description` written as "the caller
  ..." (the decision model reads it), and the code that runs. It fires and becomes an option for Winnow automatically.
- **Real data:** put `customers.csv`, `location_notes.csv` (and optionally `disputes.csv`, `fleet.csv`,
  `airport_aliases.csv`) in a folder, mount it at `/reference` (uncomment the volume in `docker-compose.yml`), set
  `REFERENCE_DIR=/reference`. The samples are in `reference/`.
- **Tail numbers and customer names:** set `EXTRACT_MODEL_URL` (and `EXTRACT_MODEL` for an OpenAI-compatible
  endpoint) to the model that pulls them out. On each caller line the same Winnow request asks "did the caller just say
  a complete tail number?" (when the line is about a tail, or answers the agent asking for one) and "did they name
  their company?" (until the customer is known, or when the line sounds like a new name). On yes, that line goes to the
  model. A tail must then be a valid registration and, if the customer has a fleet on file, one of their tails; a name
  must match a customer in `customers.csv`. Either way the agent is prompted to read it back. Account numbers said as
  digits are still matched directly. Without `EXTRACT_MODEL_URL`, or with `EXTRACT_MODE=parser`, the built-in list
  matching and tail parser are used instead (see Notes). See `decisions/model_extract.py`.
- **How sure Winnow must be:** `DECIDE_MIN_P` (default 0.5).
- **Fuzzy names:** customer and location are matched against the reference lists first. When a match came from a heard
  name rather than an exact code or account number, the same Winnow request also asks "which customer / airport did
  they mean?" over the closest few names plus "none of these", and its pick settles the variable (functions re-run if
  it changed). The agent is also prompted to read the name back. Each decision prints its round-trip time (`ms`).

| File | What it does |
|---|---|
| `decisions/events.py` | Nova Sonic / plain events -> (speaker, text, final) |
| `decisions/engine.py` | one call: variables, auto-calling, decisions |
| `decisions/functions.py` | the functions and the variables they need |
| `decisions/decider.py` | Winnow-12B questions on `/v1/systemone` (intent, "was a tail / name said?", tie-breaks), keyword fallback |
| `decisions/model_extract.py` | the external model (GPT 6 Luna) that pulls out tail numbers and customer names |
| `decisions/extract.py`, `slots.py`, `reference.py` | list matching, the tail parser, and the reference CSV loader |
| `decisions/server.py` | the HTTP API |
| `decisions/replay.py` | feed a demo call, Nova events or typed lines |
| `decisions/test_decisions.py` | tests (`python -m unittest decisions.test_decisions`) |
| `reference/`, `demos/` | sample customers / locations / fleet, and three demo calls |
| `winnow/serve.py` | starts winnow-server inside its container |
| `scripts/fetch_model.sh` | checks the model file, or downloads it (`MODELS_SOURCE`) |
| `scripts/publish_model.py` | optional: put the model on a GitHub release in parts |

## GPT 6 Luna on Bedrock for tails and names

GPT 6 Luna is on Amazon Bedrock with an OpenAI-compatible Chat Completions endpoint, so the extract model needs no extra
code: point the settings below at it. The same model and settings serve tail numbers and customer names. (Sources: the Bedrock
[GPT-6 Luna model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-6-luna.html),
[Chat Completions on Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions.html),
[API key permissions](https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys-permissions.html).)

**1. Pick the inference profile.** Luna can only be called through a cross-Region inference profile, never by its
base model ID on `bedrock-runtime`. AWS already defines two; you don't create them:

| Profile ID | Regions you can call it from |
|---|---|
| `us.openai.gpt-6-luna` | us-east-1, us-east-2, us-west-1, us-west-2, ca-central-1, ca-west-1 |
| `global.openai.gpt-6-luna` | those plus most EU / AP / SA Regions |

Use `us.openai.gpt-6-luna` from a US Region (same Region as the EC2 instance keeps latency down). Want your own profile
for cost tracking? Create an *application inference profile* from the `us.` one (Bedrock console → Cross-Region
inference, or `aws bedrock create-inference-profile`) and use its ARN as the model; that variant is untested here.

**2. Allow it in IAM.** The identity behind the API key needs to invoke the profile and the model in each Region the
profile routes to, and to use bearer-token (API key) auth:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect": "Allow",
     "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
     "Resource": ["arn:aws:bedrock:*:<account-id>:inference-profile/us.openai.gpt-6-luna",
                  "arn:aws:bedrock:*::foundation-model/openai.gpt-6-luna"]},
    {"Effect": "Allow", "Action": "bedrock:CallWithBearerToken", "Resource": "*"}
  ]
}
```

The first call to a third-party model may also need someone with AWS Marketplace permissions to accept its terms in the
Bedrock console (Model catalog → GPT-6 Luna).

**3. Get a Bedrock API key.** Bedrock console → API keys → generate a key (a long-term one is simplest for a hackathon;
give it an expiry). It goes in `.env` only; `.env` is git-ignored.

**4. Set `.env`** and restart (`docker compose up -d`):

```
EXTRACT_MODE=model
EXTRACT_MODEL_URL=https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1
EXTRACT_MODEL=us.openai.gpt-6-luna
EXTRACT_MODEL_API_KEY=<your Bedrock API key>
EXTRACT_MODEL_REASONING=none
EXTRACT_MODEL_TIMEOUT=3
```

Change the Region in the URL to yours. `EXTRACT_MODEL_REASONING=none` asks Luna not to reason first (its default is
`medium`), which is what keeps each call fast; the request sends `max_completion_tokens` and no temperature.

**5. Check it** without a call in progress:

```bash
curl -s https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/chat/completions \
  -H "Authorization: Bearer $EXTRACT_MODEL_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"us.openai.gpt-6-luna","reasoning_effort":"none","max_completion_tokens":16,
       "messages":[{"role":"user","content":"Write this aircraft registration with no dashes: November eight eight November whiskey"}]}'
```

Then send `caller: Hi, this is Dana with Northwind Aviation` and `caller: tail number November eight eight November
whiskey` (with `python -m decisions.replay --url http://localhost:8000 -`): the variables fill and the agent gets "Confirm the account" and "Read the tail number back". If the call fails, the API logs Bedrock's reason (`docker compose logs api`), and that
variable is simply not filled from that line.

Not done here: SigV4 signing with the EC2 instance role instead of an API key (the API container is standard-library
only); `bedrock-mantle` (us-east-1 only, `.../openai/v1` on `bedrock-mantle.us-east-1.api.aws`, base model ID
`openai.gpt-6-luna`) should also work with the same settings but wasn't tried. None of this was run against Bedrock
from here; it follows the AWS docs above.

## Notes: omitted for now

- **Tail numbers and customer names go to the external model, not the parser.** The deterministic tail parser
  (`decisions/slots.py`: phonetic alphabet, aviation digits, "one twenty three", a dropped "November", tails over several
  breaths) and the fuzzy customer-name matcher (`decisions/extract.py`: "Blue Herring" -> Blue Heron) are not the main
  path. They are used only with `EXTRACT_MODE=parser` or when no `EXTRACT_MODEL_URL` is set. On the main path the
  matcher only checks the model's answer against `customers.csv`; account numbers said as digits still match directly.
- **No special handling of mid-tail corrections** ("forget the November, I meant tango", "scratch that") on the main
  path: the model sees the caller's line and the few before it and is trusted to resolve them; the format, fleet and
  customer-list checks and the agent's read-back are the safety net. (The parser handles them when it's in use, and
  its tests stay.)
- **Locations** still come from list matching (ICAO / IATA / FBO / airport names), with Winnow settling fuzzy matches.
- **Removed from this repo:** the Django web UI and browser console, the speech pipeline (Nemotron ASR, diarization,
  WAV streaming), the old sentiment scorer and use-case tools, their tests and audio fixtures. They are in the git
  history before this change if you need them back.
