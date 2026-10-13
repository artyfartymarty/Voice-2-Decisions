#!/usr/bin/env bash
# From a fresh GPU instance to a running, checked service.
#
#   ./setup.sh               # check GPU + Docker, write .env, get Winnow-12B from GitHub, build, start, smoke test
#   ./setup.sh --skip-build  # after a reboot: reuse the built images
#
# Needs: NVIDIA driver 580+, Docker with the NVIDIA Container Toolkit (the AWS "Deep Learning Base OSS Nvidia Driver
# GPU AMI (Ubuntu 22.04)" has all three), ~60 GB free disk.
set -euo pipefail
cd "$(dirname "$0")"
step() { printf '\n==> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

step "GPU and driver"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: use the Deep Learning Base OSS Nvidia Driver GPU AMI"
IFS=',' read -r gpu driver cap mem < <(nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader,nounits | head -1)
echo "GPU${gpu}, driver${driver}, compute capability${cap}, ${mem// /} MiB"
driver="${driver// /}"; cap="${cap// /}"; mem="${mem// /}"
[ "${driver%%.*}" -ge 580 ] || die "driver $driver is older than 580 (Winnow's CUDA 13 build needs 580+); use the Deep Learning Base OSS Nvidia Driver AMI"
[ "$mem" -ge 15000 ] || die "${mem} MiB of GPU memory; Winnow-12B Q8 needs ~14 GB. Use a g5 / g6 (24 GB)."
arch="${cap/./}"

step "Docker"
command -v docker >/dev/null || die "docker not found"
docker compose version >/dev/null || die "docker compose plugin not found"
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi -L >/dev/null \
  || die "Docker can't see the GPU: sudo apt-get install -y nvidia-container-toolkit && sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"

step ".env"
if [ ! -f .env ]; then cp .env.example .env; echo "wrote .env"; fi
sed -i "s|^CUDA_ARCH=.*|CUDA_ARCH=$arch|" .env
echo "CUDA_ARCH=$arch"

step "Winnow-12B from GitHub (12.7 GB the first time, verified by SHA-256)"
scripts/fetch_model.sh

if [ "${1:-}" != "--skip-build" ]; then
  step "Build (Winnow compiles CUDA kernels for sm_$arch; the first build takes a while)"
  docker compose build
fi

step "Start"
docker compose up -d
port=$(grep -E '^PORT=' .env | cut -d= -f2); port=${port:-8000}
printf 'waiting for the API'
for _ in $(seq 1 60); do curl -fs "http://localhost:$port/healthz" >/dev/null 2>&1 && break; printf '.'; sleep 2; done
echo
printf 'waiting for Winnow-12B to load'
for _ in $(seq 1 120); do
  docker compose exec -T winnow python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8091/health', timeout=2)" >/dev/null 2>&1 && break
  printf '.'; sleep 5
done
echo
docker compose ps

step "Smoke test: replay the demo quote call through the API"
docker compose exec -T api python -m decisions.replay --url http://localhost:8000 --call smoke tests/fixtures/demo_quote_call.script.json

cat <<EOF

Ready. Decisions above should say [winnow, p=...]; [rules ...] means Winnow isn't answering yet
(docker compose logs winnow).

From your laptop:
  ssh -L $port:localhost:$port ubuntu@<this-instance>
then open http://localhost:$port for the console, or POST transcript events to
  http://localhost:$port/calls/<call-id>/events
EOF
