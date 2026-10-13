#!/usr/bin/env bash
# One command from a fresh GPU instance to a running, checked stack.
#
#   ./scripts/setup.sh            # check GPU + driver, write .env, fetch models, build, start, run checks
#   ./scripts/setup.sh --skip-build   # reuse built images (e.g. after a reboot)
#
# Needs: NVIDIA driver 580+, Docker with the NVIDIA Container Toolkit (the AWS "Deep Learning Base OSS Nvidia
# Driver GPU AMI" has all three), curl, ~80 GB free disk.
set -euo pipefail
cd "$(dirname "$0")/.."
step() { printf '\n==> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

SKIP_BUILD=0
[ "${1:-}" = "--skip-build" ] && SKIP_BUILD=1

step "GPU and driver"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: use the Deep Learning Base OSS Nvidia Driver GPU AMI"
read -r gpu driver cap < <(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader | head -1 | awk -F', ' '{gsub(/ /,"_",$1); print $1, $2, $3}')
echo "GPU $gpu, driver $driver, compute capability $cap"
[ "${driver%%.*}" -ge 580 ] || die "driver $driver is older than 580 (Winnow's CUDA 13 build needs 580+). The ECS GPU AMI ships 550; use the Deep Learning Base AMI."
arch="${cap/./}"
mem_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
[ "$mem_mib" -ge 22000 ] || echo "WARNING: ${mem_mib} MiB of GPU memory; Winnow-12B + speech needs ~16 GB free. 24 GB cards (g5/g6) are the target."

step "Docker"
command -v docker >/dev/null || die "docker not found"
docker compose version >/dev/null || die "docker compose plugin not found"
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi -L >/dev/null \
  || die "Docker can't see the GPU: sudo apt-get install -y nvidia-container-toolkit && sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
echo "Docker sees the GPU"

step ".env"
if [ ! -f .env ]; then
  secret=$(python3 -c 'import secrets; print(secrets.token_urlsafe(50))')
  sed -e "s|^DJANGO_SECRET_KEY=.*|DJANGO_SECRET_KEY=$secret|" -e "s|^CUDA_ARCH=.*|CUDA_ARCH=$arch|" .env.example > .env
  echo "wrote .env (CUDA_ARCH=$arch)"
else
  sed -i "s|^CUDA_ARCH=.*|CUDA_ARCH=$arch|" .env
  echo "kept existing .env, set CUDA_ARCH=$arch"
fi

step "Models (~13.9 GB the first time; verified by SHA-256)"
scripts/fetch_models.sh

if [ "$SKIP_BUILD" = 0 ]; then
  step "Build images (Winnow compiles for sm_$arch; the first build takes several minutes)"
  docker compose build
fi

step "Start"
docker compose up -d
printf 'waiting for the web app'
for _ in $(seq 1 120); do
  if curl -fs "http://localhost:$(grep -E '^PORT=' .env | cut -d= -f2 || echo 8080)/healthz" >/dev/null 2>&1; then break; fi
  printf '.'; sleep 5
done
echo
docker compose ps

step "Checks on the real models"
docker compose exec -T web python scripts/check.py --stack

port=$(grep -E '^PORT=' .env | cut -d= -f2); port=${port:-8080}
cat <<EOF

Ready. From your laptop:
  ssh -L $port:localhost:$port ubuntu@<this-instance>
then open http://localhost:$port and press a test call (or drop a WAV file).

Stream a WAV in real time from the instance:
  docker compose exec web python scripts/stream_wav.py tests/fixtures/quote_call_phone_mono.wav --chunk 2
EOF
