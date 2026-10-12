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
