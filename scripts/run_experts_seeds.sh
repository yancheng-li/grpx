#!/usr/bin/env bash
# 专家层三种子正式训练（与注意力组设置对齐）。
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p outputs

for seed in 0 1 2; do
  echo "===== START experts seed=${seed} $(date -Is) ====="
  grpx train \
    --config configs/arithmetic_olmoe_d2d3_experts.toml \
    --set "training.seed=${seed}" \
    --set "training.run_name=olmoe-arith-d2d3-experts-s${seed}" \
    --set training.max_rows_per_minibatch=128 \
    --set training.max_tokens_per_minibatch=8192 \
    2>&1 | tee "outputs/olmoe_d2d3_experts_s${seed}.log"
  echo "===== DONE experts seed=${seed} $(date -Is) ====="
done

echo "===== ALL EXPERT SEEDS DONE $(date -Is) ====="
