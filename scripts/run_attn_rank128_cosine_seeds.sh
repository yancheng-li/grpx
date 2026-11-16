#!/usr/bin/env bash
# 注意力秩 128 + cosine，150 轮 × 3 seeds；跑完后 500 题评测。
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p outputs

for seed in 0 1 2; do
  echo "===== START rank128-cosine seed=${seed} $(date -Is) ====="
  grpx train --config configs/arithmetic_olmoe_d2d3_attn_rank128_cosine.toml \
    --set "training.seed=${seed}" \
    --set "training.run_name=olmoe-arith-d2d3-attn-rank128-cosine-s${seed}" \
    2>&1 | tee "outputs/olmoe_d2d3_attn_rank128_cosine_s${seed}.log"
  echo "===== DONE rank128-cosine seed=${seed} $(date -Is) ====="
done

PYTHONPATH=. python scripts/eval_d3_500.py \
  --checkpoints \
    outputs/olmoe-arith-d2d3-attn-rank128-cosine-s0/checkpoint-0150 \
    outputs/olmoe-arith-d2d3-attn-rank128-cosine-s1/checkpoint-0150 \
    outputs/olmoe-arith-d2d3-attn-rank128-cosine-s2/checkpoint-0150 \
  --output outputs/olmoe_d2d3_attn_rank128_cosine_150_d3_500_eval.json \
  2>&1 | tee outputs/olmoe_d2d3_attn_rank128_cosine_150_d3_500_eval.log

echo "########## RANK128 COSINE DONE $(date -Is) ##########"
