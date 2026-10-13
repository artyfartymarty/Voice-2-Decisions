#!/bin/bash
# Boot script for the ECS GPU instance: put Winnow-12B on the host's disk (not in the image), then join the cluster.
# scripts/ecs_deploy.sh fills in CLUSTER, MODEL_S3_URI and MODEL_SHA256.
set -euo pipefail
command -v aws >/dev/null || dnf install -y awscli 2>/dev/null || yum install -y awscli
dir=/opt/voice2decisions/models/winnow/gguf
file=$dir/Winnow-12B-Q8_0.gguf
mkdir -p "$dir"
if [ ! -f "$file" ] || [ "$(sha256sum "$file" | cut -d' ' -f1)" != "${MODEL_SHA256}" ]; then
  aws s3 cp --only-show-errors "${MODEL_S3_URI}" "$file.part"
  [ "$(sha256sum "$file.part" | cut -d' ' -f1)" = "${MODEL_SHA256}" ] || { echo "model checksum mismatch" >&2; exit 1; }
  mv "$file.part" "$file"
fi
# Join the cluster only once the model is on disk, so the task never starts without it.
cat >> /etc/ecs/ecs.config <<CFG
ECS_CLUSTER=${CLUSTER}
ECS_ENABLE_GPU_SUPPORT=true
ECS_ENABLE_CONTAINER_METADATA=true
CFG
# Winnow's CUDA 13 build needs NVIDIA driver 580 or newer; this lands in /var/log/cloud-init-output.log.
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
