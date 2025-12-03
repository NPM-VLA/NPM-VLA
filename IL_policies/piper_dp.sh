#!/usr/bin/env bash
set -e

export HF_DATASET_REPO="Anlorla/sweep2E"
export LEROBOT_DATA_ROOT="/home/jovyan/.cache/huggingface/lerobot/zeno/sweep2E_v1"

python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="${HF_DATASET_REPO}" \
  --dataset.root="${LEROBOT_DATA_ROOT}" \
  --policy.type=diffusion \
  --output_dir "/home/jovyan/workspace/IL_policies/checkpoints/sweep2E_dp" \
  --job_name="diffusion_sweep2E" \
  --policy.device=cuda \
  --batch_size=16 \
  --steps=80000 \
  --log_freq=100 \
  --eval_freq=5000 \
  --save_freq=5000 \
  --wandb.enable=true \
  --policy.repo_id=false