#!/usr/bin/env bash
# On the GPU instance: pull the private images from ECR with the instance role and start them. No source code needed.
#
#   scripts/ecr_deploy.sh
#
# Needs in this folder: docker-compose.ecr.yml, .env (ECR_REGISTRY, IMAGE_TAG, AWS_REGION, ...; see .env.example)
# and models/winnow/gguf/Winnow-12B-Q8_0.gguf. The instance role needs ECR pull and Parameter Store read
# (README: Private images on ECR).
set -euo pipefail
cd "$(dirname "$0")/.."
step() { printf '\n==> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
[ -f .env ] || die "no .env: copy .env.example and set ECR_REGISTRY, IMAGE_TAG and AWS_REGION"
set -a; . ./.env; set +a
[ -n "${ECR_REGISTRY:-}" ] || die "set ECR_REGISTRY in .env (scripts/ecr_push.sh prints it)"
[ -f models/winnow/gguf/Winnow-12B-Q8_0.gguf ] || die "models/winnow/gguf/Winnow-12B-Q8_0.gguf is missing (README: Get the model onto the instance)"
region=${AWS_REGION:?set AWS_REGION in .env}
compose() { docker compose -f docker-compose.ecr.yml "$@"; }

step "Log in to $ECR_REGISTRY with the instance role"
aws ecr get-login-password --region "$region" | docker login --username AWS --password-stdin "$ECR_REGISTRY"

step "Parameter Store ${SSM_PARAMETER_PATH:-/voice2decisions/} (names only)"
aws ssm get-parameters-by-path --region "$region" --path "${SSM_PARAMETER_PATH:-/voice2decisions/}" --recursive \
  --query 'Parameters[].Name' --output text || die "the instance role can't read Parameter Store"

step "Pull and start ${IMAGE_TAG:-latest}"
compose pull
compose up -d
port=${PORT:-8000}
printf 'waiting for the API'
for _ in $(seq 1 60); do curl -fs "http://localhost:$port/healthz" >/dev/null 2>&1 && break; printf '.'; sleep 2; done
echo
curl -fs "http://localhost:$port/healthz" >/dev/null || { compose logs --tail 30 api; die "the API didn't start (log above)"; }
printf 'waiting for Winnow-12B to load'
for _ in $(seq 1 120); do
  compose exec -T winnow python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8091/health', timeout=2)" >/dev/null 2>&1 && break
  printf '.'; sleep 5
done
echo
compose ps

step "Smoke test: stream the demo quote call through the API"
compose exec -T api python -m decisions.simulate --speed 0 --url http://localhost:8000 demos/demo_quote_call.json
