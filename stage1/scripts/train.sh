#!/usr/bin/env bash
# Usage: GPUS=0,1,2,3 scripts/train.sh CONFIG [hae.train arguments]
#   GPUS   comma-separated GPU ids (default 0); global_batch_size must be a multiple of micro batch x GPUs
#   MICRO  images per GPU per forward pass (default: training.micro_batch_size of the config)
# Examples:
#   GPUS=0,1,2,3 scripts/train.sh configs/stage1_alignment.yaml
#   GPUS=0,1,2,3 scripts/train.sh configs/stage1_alignment.yaml --resume runs/stage1_alignment/checkpoint_last.pt
#   GPUS=0,1,2,3 MICRO=32 scripts/train.sh configs/stage2_gan.yaml --set logging.wandb.mode=offline
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
config="${1:?Usage: GPUS=0,1 scripts/train.sh CONFIG [--resume CKPT | --init-from CKPT | --continue-from CKPT] [--set KEY=VALUE ...]}"
shift
gpus="${GPUS:-0}"
n_gpus=$(awk -F, '{print NF}' <<< "$gpus")
micro=()
if [ -n "${MICRO:-}" ]; then micro=(--set "training.micro_batch_size=$MICRO"); fi
export CUDA_VISIBLE_DEVICES="$gpus"
exec torchrun --standalone --nproc_per_node="$n_gpus" -m hae.train --config "$config" ${micro[@]+"${micro[@]}"} "$@"
