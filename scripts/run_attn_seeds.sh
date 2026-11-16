#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/projects/grpx
source /root/miniconda3/etc/profile.d/conda.sh
conda activate grpx
export PYTHONPATH=.

for seed in 0 1 2; do
  echo "===== START seed=${seed} $(date -Is) ====="
  grpx train \
    --config configs/arithmetic_olmoe_d2d3.toml \
    --set "training.seed=${seed}" \
    --set "training.run_name=olmoe-arith-d2d3-attn-s${seed}" \
    2>&1 | tee "outputs/olmoe_d2d3_attn_s${seed}.log"
  echo "===== DONE seed=${seed} $(date -Is) ====="
done

echo "===== ALL SEEDS DONE $(date -Is) ====="
python scripts/_summarize_attn_evals.py
