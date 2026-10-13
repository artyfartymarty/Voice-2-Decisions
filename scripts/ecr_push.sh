#!/usr/bin/env bash
# Build both images and push them to private ECR repositories, so the code ships as images and never as source.
#
#   AWS_REGION=us-east-1 PULL_ROLE_ARNS=arn:aws:iam::123456789012:role/v2d-ec2 scripts/ecr_push.sh
#
# Run it where you have Docker and AWS credentials allowed to create and push to ECR (your laptop, or the GPU
# instance with a build role). It creates <ECR_PREFIX>/api and <ECR_PREFIX>/winnow if missing (private, scanned on
# push, encrypted), sets their repository policy so only PULL_ROLE_ARNS (and you) can pull or push, then builds,
# tags with IMAGE_TAG and "latest", and pushes. Prints the ECR_REGISTRY and IMAGE_TAG for the instance's .env.
#
# Settings (environment or .env): AWS_REGION, ECR_PREFIX (default voice2decisions), IMAGE_TAG (default the git
# commit), PULL_ROLE_ARNS (comma-separated IAM role ARNs; empty = leave the repository policy alone), CUDA_ARCH.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a
step() { printf '\n==> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
command -v aws >/dev/null || die "aws CLI not found"
command -v docker >/dev/null || die "docker not found"

region=${AWS_REGION:-$(aws configure get region || true)}
[ -n "$region" ] || die "set AWS_REGION"
prefix=${ECR_PREFIX:-voice2decisions}
tag=${IMAGE_TAG:-$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M)}
account=$(aws sts get-caller-identity --query Account --output text)
me=$(aws sts get-caller-identity --query Arn --output text)
# An assumed-role session is matched by its role's ARN in policies.
me=$(printf '%s' "$me" | sed -E 's#^arn:aws:sts::([0-9]+):assumed-role/([^/]+)/.*#arn:aws:iam::\1:role/\2#')
registry="$account.dkr.ecr.$region.amazonaws.com"

policy() {  # only these principals may pull or push, whatever other IAM policies in the account allow
  local arns=$1
  python3 - "$arns" <<'PY'
import json, sys
arns = sorted({a.strip() for a in sys.argv[1].split(",") if a.strip()})
pull = ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"]
push = ["ecr:PutImage", "ecr:InitiateLayerUpload", "ecr:UploadLayerPart", "ecr:CompleteLayerUpload"]
print(json.dumps({"Version": "2012-10-17", "Statement": [
    {"Sid": "TeamPull", "Effect": "Allow", "Principal": {"AWS": arns}, "Action": pull},
    {"Sid": "OnlyTheTeam", "Effect": "Deny", "Principal": "*", "Action": pull + push,
     "Condition": {"ArnNotEquals": {"aws:PrincipalArn": arns}}},
]}))
PY
}

for repo in api winnow; do
  step "ECR repository $prefix/$repo"
  if ! aws ecr describe-repositories --region "$region" --repository-names "$prefix/$repo" >/dev/null 2>&1; then
    aws ecr create-repository --region "$region" --repository-name "$prefix/$repo" \
      --image-scanning-configuration scanOnPush=true --encryption-configuration encryptionType=KMS >/dev/null
    echo "created"
  fi
  if [ -n "${PULL_ROLE_ARNS:-}" ]; then
    aws ecr set-repository-policy --region "$region" --repository-name "$prefix/$repo" \
      --policy-text "$(policy "$PULL_ROLE_ARNS,$me")" >/dev/null
    echo "pull/push limited to: $PULL_ROLE_ARNS,$me"
  fi
done

step "Log in to $registry"
aws ecr get-login-password --region "$region" | docker login --username AWS --password-stdin "$registry"

step "Build (Winnow compiles CUDA kernels for sm_${CUDA_ARCH:-86}; the first build takes a while)"
docker compose build
docker build -t text2decisions-winnow:local --build-arg BASE=winnow-inference:a50c7d0 winnow

step "Push tag $tag"
for pair in "text2decisions:latest api" "text2decisions-winnow:local winnow"; do
  set -- $pair
  for t in "$tag" latest; do
    docker tag "$1" "$registry/$prefix/$2:$t"
    docker push "$registry/$prefix/$2:$t"
  done
done

cat <<MSG

Pushed. On the instance, put these in .env and run scripts/ecr_deploy.sh:
  ECR_REGISTRY=$registry
  ECR_PREFIX=$prefix
  IMAGE_TAG=$tag
  AWS_REGION=$region
MSG
