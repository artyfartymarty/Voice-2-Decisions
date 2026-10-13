# Voice-2-Decisions

Nova Sonic turns the call's speech into text. This service takes that text and does the rest:

> **Speech-to-text is no longer in this repo.** It moved to Amazon Nova Sonic, which runs on the UI side. The Nemotron
> ASR model, its container, its download and its setup steps are removed. The old Nemotron setup lives in the separate
> Voice-Nemotron-setup repo.

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
              "say": "Quote NORTHWIND AVIATION, LLC at KTEB: current 0.3125 GPPG. Don't go below the minimum, 0.0850 GPPG (0.2275 of room).",
              "result": {"current_gppg": 0.3125, "floor_gppg": 0.085, "...": "..."},
              "because": "Northwind, account 204511. Price per gallon at Teterboro?"},
 "variables": {"customer_number": "204511", "icao": "KTEB"}}
```

Other endpoints: `GET /calls/<id>` (full call state), `DELETE /calls/<id>`, `GET /functions`, `GET /healthz`.
Set `DECISION_WEBHOOK_URL` to have every new decision POSTed to your app as well.

## Simulate the Nova Sonic side

`decisions/simulate.py` stands in for your boss's end. It streams a scripted call into the service the way Nova Sonic
does: for each turn, a speculative text block that grows a few words at a time while the person talks, then the final
text (`contentStart`, `textOutput`, `contentEnd`, `USER` for the caller and `ASSISTANT` for the agent), at the call's
real pace. The variables, function calls and decisions print as they happen.

```bash
docker compose exec api python -m decisions.simulate --url http://localhost:8000 demos/demo_quote_call.json
python -m decisions.simulate --url http://<ec2-host>:8000 demos/*.json       # from your laptop, all three calls
python -m decisions.simulate --speed 4 demos/demo_frustrated_caller.json     # in-process, no server, 4x speed
python -m decisions.simulate --dump demos/demo_quote_call.json > call.jsonl  # just the events, to hand to the UI side
```

`--speed 0` sends without waiting, `--chunk N` sets the words per speculative update, and `--no-partials` sends only
final text. To add a call, put a new JSON file in `demos/`: a list of `{"speaker": "caller" | "agent", "text": "..."}`
turns. `start` and `end` (seconds) are optional; without them each turn is timed from its word count.

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
- **Real data:** see "Use your own customers file" below. The samples are in `reference/`.
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
- **Misheard names and tails (main path):** when the name GPT 6 Luna pulls out isn't a customer exactly (the full name,
  or a first word only one customer has), the customers that look or sound closest go to Winnow as one choice question
  with the caller's line and "none of these". For "Arigo" that's Arrigo Aviation, Arrow Jet Charter and Aero Vista;
  the sound check uses a phonetic key, so Arrigo / Arrow / Aero land together. Winnow's pick is read back to the
  caller; "none" (or a pick below `DECIDE_MIN_P`) asks for the account number instead. A tail that isn't on the
  customer's fleet works the same way, with the fleet's tails one or two characters off ("N47JA": N47AJ, N74AJ).
  Winnow gets only text: Nova Sonic's transcript, never the audio.
- **A tail on file identifies the customer.** If the tail is on a customer's fleet (`fleet.csv`), that customer is
  used, over a name that was heard; when they disagree ("Arrow" heard, tail on Aero Vista's fleet) the agent is told
  to read the account back. An account number said as digits, written or spoken ("two oh four five one one"), fills
  the customer directly.

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
| `decisions/simulate.py` | stream a scripted call like Nova Sonic (partials, then final text, in real time) |
| `decisions/test_decisions.py` | tests (`python -m unittest decisions.test_decisions`) |
| `reference/`, `demos/` | sample customers / locations / fleet, and three demo calls |
| `winnow/serve.py` | starts winnow-server inside its container |
| `scripts/fetch_model.sh` | checks the model file, or downloads it (`MODELS_SOURCE`) |
| `scripts/publish_model.py` | optional: put the model on a GitHub release in parts |
| `scripts/ecr_push.sh`, `scripts/ecr_deploy.sh`, `docker-compose.ecr.yml`, `winnow/Dockerfile` | private ECR images: build and push, then pull and run on the instance |
| `decisions/ssm.py` | reads secrets from Parameter Store when the container starts |
| `ecs/`, `scripts/ecs_deploy.sh` | the 24/7 ECS service: task definition, instance boot script, IAM, deploy |

## Private images on ECR (keep the code from other teams)

The code ships as Docker images in private ECR repositories, so the instance never holds a git clone, and secrets live
in Parameter Store, not in the image, `.env` or git. Both images carry everything they run (the API's Python and
reference data; Winnow's start wrapper). Only the model file is mounted.

**1. Build and push** from anywhere with Docker and AWS credentials that can create and push to ECR (your laptop, or
the GPU instance before you remove the source):

```bash
AWS_REGION=us-east-1 PULL_ROLE_ARNS=arn:aws:iam::<account>:role/<instance-role> scripts/ecr_push.sh
```

It creates `voice2decisions/api` and `voice2decisions/winnow` (private, scanned on push, KMS-encrypted), builds both
images, tags them with the git commit and `latest`, pushes, and prints the four lines for the instance's `.env`.
`PULL_ROLE_ARNS` is a comma-separated list of the roles your team uses. The script sets this repository policy, so only
those roles and you can pull or push, whatever other IAM policies in the account allow:

```json
{"Version": "2012-10-17", "Statement": [
  {"Sid": "TeamPull", "Effect": "Allow", "Principal": {"AWS": ["arn:aws:iam::<account>:role/<instance-role>", "<you>"]},
   "Action": ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"]},
  {"Sid": "OnlyTheTeam", "Effect": "Deny", "Principal": "*",
   "Action": ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability",
              "ecr:PutImage", "ecr:InitiateLayerUpload", "ecr:UploadLayerPart", "ecr:CompleteLayerUpload"],
   "Condition": {"ArnNotEquals": {"aws:PrincipalArn": ["arn:aws:iam::<account>:role/<instance-role>", "<you>"]}}}]}
```

**2. Put the secrets in Parameter Store.** Every parameter under `/voice2decisions/` becomes the environment variable
of the same name when the API container starts (`SSM_PARAMETER_PATH` to change the path). Use `SecureString` for
anything secret. Non-secret settings can go there too, or stay in `.env`; a value set in `.env` wins.

```bash
aws ssm put-parameter --region us-east-1 --type SecureString --name /voice2decisions/EXTRACT_MODEL_API_KEY --value '<Bedrock API key>'
aws ssm put-parameter --region us-east-1 --type String --name /voice2decisions/EXTRACT_MODEL_URL \
  --value https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1
```

**3. Give the instance a role** (EC2 console: Actions, Security, Modify IAM role) with this policy. Replace the region,
account and, if you encrypt the parameters with your own KMS key, the key ARN (`*` here covers the default
`aws/ssm` key, limited to use through Parameter Store):

```json
{"Version": "2012-10-17", "Statement": [
  {"Sid": "EcrLogin", "Effect": "Allow", "Action": "ecr:GetAuthorizationToken", "Resource": "*"},
  {"Sid": "EcrPull", "Effect": "Allow",
   "Action": ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"],
   "Resource": "arn:aws:ecr:us-east-1:<account>:repository/voice2decisions/*"},
  {"Sid": "Secrets", "Effect": "Allow", "Action": ["ssm:GetParametersByPath", "ssm:GetParameter", "ssm:GetParameters"],
   "Resource": ["arn:aws:ssm:us-east-1:<account>:parameter/voice2decisions", "arn:aws:ssm:us-east-1:<account>:parameter/voice2decisions/*"]},
  {"Sid": "DecryptSecureStrings", "Effect": "Allow", "Action": "kms:Decrypt", "Resource": "*",
   "Condition": {"StringEquals": {"kms:ViaService": "ssm.us-east-1.amazonaws.com"}}}]}
```

The API container gets the role's credentials from the instance metadata service, which by default answers only one
network hop away. Allow two so containers can reach it:

```bash
aws ec2 modify-instance-metadata-options --instance-id <id> --http-tokens required --http-put-response-hop-limit 2
```

**4. On the instance** you need only `docker-compose.ecr.yml`, `scripts/ecr_deploy.sh`, a `.env` and the model file:

```bash
mkdir -p ~/v2d/scripts ~/v2d/models/winnow/gguf && cd ~/v2d
# copy in docker-compose.ecr.yml, scripts/ecr_deploy.sh and the model (Get the model onto the instance), then:
cat > .env <<'ENV'
ECR_REGISTRY=<account>.dkr.ecr.us-east-1.amazonaws.com
ECR_PREFIX=voice2decisions
IMAGE_TAG=<printed by ecr_push.sh>
AWS_REGION=us-east-1
ENV
scripts/ecr_deploy.sh
```

It logs in to ECR with the instance role, lists the parameter names it can read, pulls, starts and streams the demo
quote call through the API. To ship a change, run `scripts/ecr_push.sh` again and `scripts/ecr_deploy.sh` with the new
`IMAGE_TAG`. If the API won't start, `docker compose -f docker-compose.ecr.yml logs api` says which permission or
parameter is missing.

Private images keep the code away from anyone who can't pull them. Anyone who can pull the image, or who has a shell on
the instance, can still read the Python inside it, so keep both limited to your team.

## Run it 24/7 on ECS

For the hackathon, one EC2 instance with docker compose (above) is quickest. For an always-on service, run it as an ECS
service with one always-running task, so ECS restarts it if it fails and replaces the instance if that fails.

**How it's laid out.** Winnow-12B needs a GPU and Fargate has none, so it runs on the ECS EC2 launch type: one
`g5.2xlarge` from the ECS GPU-optimized AMI in an Auto Scaling group, attached to the cluster as a capacity provider.
The API runs in the **same task** as Winnow, which is the simpler option: one task to deploy and watch, host
networking so the API reaches Winnow on `127.0.0.1:8091`, and no service discovery or load balancer. The task asks for
one GPU, the service keeps one copy running, and both containers have health checks. Images come from the private ECR
repositories. The 12.7 GB model is not in the image: the instance copies it from S3 to its own disk on first boot,
checks its SHA-256, and only then joins the cluster. Secrets go in the task definition's `secrets`, one per Parameter
Store parameter under `/voice2decisions/`, so ECS injects them and `ssm.py` isn't involved.

| File | What it is |
|---|---|
| `ecs/task-definition.json` | the task: `winnow` (1 GPU, model volume) and `api` (port 8000, secrets), health checks, CloudWatch logs |
| `ecs/user-data.sh` | instance boot: copy and check the model, then join the cluster with GPU support on |
| `ecs/execution-role-policy.json` | extra permissions for the task execution role |
| `ecs/instance-role-policy.json` | extra permissions for the instance role |
| `scripts/ecs_deploy.sh` | creates or updates all of it |

**Steps**, from a machine with AWS credentials that can manage IAM, EC2, Auto Scaling and ECS:

```bash
scripts/ecr_push.sh                                            # images in ECR (Private images on ECR)
aws s3 cp Winnow-12B-Q8_0.gguf s3://<bucket>/models/           # the model, once
aws ssm put-parameter --type SecureString --name /voice2decisions/EXTRACT_MODEL_API_KEY --value '<key>'   # secrets

AWS_REGION=us-east-1 ECR_REGISTRY=<account>.dkr.ecr.us-east-1.amazonaws.com IMAGE_TAG=<tag> \
MODEL_S3_URI=s3://<bucket>/models/Winnow-12B-Q8_0.gguf SUBNET_IDS=subnet-aaa,subnet-bbb ALLOWED_CIDR=<UI's range> \
scripts/ecs_deploy.sh
```

The script creates the roles, cluster, security group (port 8000 open only to `ALLOWED_CIDR`), launch template, Auto
Scaling group, capacity provider, task definition and service, and adds the two roles to the ECR repository policy.
Running it again with a new `IMAGE_TAG` registers a new task definition and rolls the service onto it. One GPU
instance holds one task, so a deploy stops the old task before starting the new one; expect a few minutes of downtime.
The first start waits for the instance to boot and copy the model. Follow it with
`aws logs tail /ecs/voice2decisions --follow`.

**IAM roles it creates:**

- `voice2decisions-ecs-execution`, assumed by `ecs-tasks.amazonaws.com`. It has `AmazonECSTaskExecutionRolePolicy`
  (ECR pull, CloudWatch logs) plus `ecs/execution-role-policy.json`: `ssm:GetParameters` on
  `parameter/voice2decisions/*` and `kms:Decrypt` through Parameter Store, for the `secrets`.
- `voice2decisions-ecs-instance` and its instance profile, assumed by `ec2.amazonaws.com`. It has
  `AmazonEC2ContainerServiceforEC2Role` (join the cluster), `AmazonSSMManagedInstanceCore` (a shell through Session
  Manager, no SSH key) and `ecs/instance-role-policy.json`: `s3:GetObject` on the model file only.
- The task itself needs no role. GPT 6 Luna is called with the Bedrock API key from Parameter Store.

**Check before relying on it:** the Winnow build needs NVIDIA driver 580 or newer. The instance logs its driver
version at boot (`/var/log/cloud-init-output.log`, or Session Manager and `nvidia-smi`). If the ECS GPU AMI's driver is
older, the `winnow` container will fail its health check and ECS will keep restarting it.

**Rough always-on cost** (us-east-1 on-demand prices as I know them; check the pricing page): `g5.2xlarge` about
$1.21 an hour, roughly $880 a month running 24/7, plus about $8 a month for the 100 GB disk and a little for logs. ECS
itself adds nothing on the EC2 launch type. A `g6.2xlarge` (L4, 24 GB) is about $0.98 an hour, roughly $715 a month,
with `INSTANCE_TYPE=g6.2xlarge` and `CUDA_ARCH=89` for the image build. A one-year Savings Plan takes roughly a third
off either.

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
give it an expiry). It goes in `.env` only; `.env` is git-ignored. When running from ECR, put it in Parameter Store
instead (see Private images on ECR).

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

### Use your own customers file

The margins file can have any name, spaces and parentheses included. It needs these column headers (any order;
other columns are ignored): `CUSTOMER_NAME`, `CUSTOMER_NUMBER`, `ICAO`, `MIN_GPPG`, `CURRENT_GPPG`.
`P12M_TOTAL_WFS_VOL` and `P12M_ADDRESSABLE_VOL` are optional. If a header is missing, the API stops at start with a
message naming it and listing the headers it found. The other files (`location_notes.csv`, and optionally
`disputes.csv`, `fleet.csv`, `airport_aliases.csv`) keep their names and sit in the same folder.

**docker compose:** copy your file into `./reference` on the instance, uncomment the `./reference:/reference:ro` volume
under `api` in `docker-compose.yml`, and set in `.env` (keep the quotes):

```
REFERENCE_DIR=/reference
CUSTOMERS_FILE="New Worksheet (1AiE8Ye4TPO)_2026-10-09-1522.csv"
```

then `docker compose up -d`. `CUSTOMERS_FILE` can also be a full path inside the container.

**ECR / ECS:** the `reference/` folder is baked into the private API image, so copy your file into `reference/` before
`scripts/ecr_push.sh`, and set the same `CUSTOMERS_FILE="..."` in `.env` (or the environment) when you run
`scripts/ecr_deploy.sh` or `scripts/ecs_deploy.sh`; the ECS script puts it in the task definition. Don't commit the
real file: it stays in your working copy and the private image only.

## Notes: omitted for now

- **Tail numbers and customer names go to the external model, not the parser.** The deterministic tail parser
  (`decisions/slots.py`: phonetic alphabet, aviation digits, "one twenty three", a dropped "November", tails over several
  breaths) and the fuzzy customer-name matcher (`decisions/extract.py`: "Blue Herring" -> Blue Heron) are not the main
  path. They are used only with `EXTRACT_MODE=parser` or when no `EXTRACT_MODEL_URL` is set. On the main path the
  model's answer is checked against `customers.csv`, and Winnow chooses among close names when it isn't exact (see
  Misheard names and tails); account numbers said as digits still match directly.
- **No special handling of mid-tail corrections** ("forget the November, I meant tango", "scratch that") on the main
  path: the model sees the caller's line and the few before it and is trusted to resolve them; the format, fleet and
  customer-list checks and the agent's read-back are the safety net. (The parser handles them when it's in use, and
  its tests stay.)
- **Locations** still come from list matching (ICAO / IATA / FBO / airport names), with Winnow settling fuzzy matches.
- **Removed from this repo:** the Django web UI and browser console, the speech pipeline (Nemotron ASR, diarization,
  WAV streaming; now Nova Sonic, and the old setup is in the Voice-Nemotron-setup repo), the old sentiment scorer and use-case tools, their tests and audio fixtures. They are in the git
  history before this change if you need them back.
