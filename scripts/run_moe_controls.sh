#!/usr/bin/env bash
# 三项对照：秩128注意力 → LR 3e-5注意力 → 两组续训到150
set -euo pipefail
cd /workspace/grpx
source .venv/bin/activate
export PYTHONPATH=/workspace/grpx
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p outputs

COMMON=(
  --config configs/arithmetic_olmoe_d2d3.toml
  --set training.max_rows_per_minibatch=128
  --set training.max_tokens_per_minibatch=8192
)

echo "########## CTRL1: attn rank=128 alpha=256 ##########"
for seed in 0 1 2; do
  echo "===== START attn-r128rank seed=${seed} $(date -Is) ====="
  grpx train "${COMMON[@]}" \
    --set "training.seed=${seed}" \
    --set "training.run_name=olmoe-arith-d2d3-attn-rank128-s${seed}" \
    --set model.lora_rank=128 \
    --set model.lora_alpha=256 \
    2>&1 | tee "outputs/olmoe_d2d3_attn_rank128_s${seed}.log"
  echo "===== DONE attn-r128rank seed=${seed} $(date -Is) ====="
done

PYTHONPATH=. python scripts/eval_d3_500.py \
  --checkpoints \
    outputs/olmoe-arith-d2d3-attn-rank128-s0/checkpoint-0060 \
    outputs/olmoe-arith-d2d3-attn-rank128-s1/checkpoint-0060 \
    outputs/olmoe-arith-d2d3-attn-rank128-s2/checkpoint-0060 \
  --output outputs/olmoe_d2d3_attn_rank128_d3_500_eval.json \
  2>&1 | tee outputs/olmoe_d2d3_attn_rank128_d3_500_eval.log

echo "########## CTRL2: attn lr=3e-5 ##########"
for seed in 0 1 2; do
  echo "===== START attn-lr3e-5 seed=${seed} $(date -Is) ====="
  grpx train "${COMMON[@]}" \
    --set "training.seed=${seed}" \
    --set "training.run_name=olmoe-arith-d2d3-attn-lr3e5-s${seed}" \
    --set training.learning_rate=3e-5 \
    2>&1 | tee "outputs/olmoe_d2d3_attn_lr3e5_s${seed}.log"
  echo "===== DONE attn-lr3e-5 seed=${seed} $(date -Is) ====="
done

PYTHONPATH=. python scripts/eval_d3_500.py \
  --checkpoints \
    outputs/olmoe-arith-d2d3-attn-lr3e5-s0/checkpoint-0060 \
    outputs/olmoe-arith-d2d3-attn-lr3e5-s1/checkpoint-0060 \
    outputs/olmoe-arith-d2d3-attn-lr3e5-s2/checkpoint-0060 \
  --output outputs/olmoe_d2d3_attn_lr3e5_d3_500_eval.json \
  2>&1 | tee outputs/olmoe_d2d3_attn_lr3e5_d3_500_eval.log

echo "########## CTRL3: continue to 150 iters ##########"
for seed in 0 1 2; do
  echo "===== START attn continue-150 seed=${seed} $(date -Is) ====="
  grpx train --resume "outputs/olmoe-arith-d2d3-attn-s${seed}-r128" \
    --set training.iterations=150 \
    2>&1 | tee -a "outputs/olmoe_d2d3_attn_s${seed}_r128_to150.log"
  echo "===== DONE attn continue-150 seed=${seed} $(date -Is) ====="
done

PYTHONPATH=. python scripts/eval_d3_500.py \
  --checkpoints \
    outputs/olmoe-arith-d2d3-attn-s0-r128/checkpoint-0150 \
    outputs/olmoe-arith-d2d3-attn-s1-r128/checkpoint-0150 \
    outputs/olmoe-arith-d2d3-attn-s2-r128/checkpoint-0150 \
  --output outputs/olmoe_d2d3_attn_r128_150_d3_500_eval.json \
  2>&1 | tee outputs/olmoe_d2d3_attn_r128_150_d3_500_eval.log

for seed in 0 1 2; do
  echo "===== START experts continue-150 seed=${seed} $(date -Is) ====="
  grpx train --resume "outputs/olmoe-arith-d2d3-experts-s${seed}" \
    --set training.iterations=150 \
    2>&1 | tee -a "outputs/olmoe_d2d3_experts_s${seed}_to150.log"
  echo "===== DONE experts continue-150 seed=${seed} $(date -Is) ====="
done

PYTHONPATH=. python scripts/eval_d3_500.py \
  --checkpoints \
    outputs/olmoe-arith-d2d3-experts-s0/checkpoint-0150 \
    outputs/olmoe-arith-d2d3-experts-s1/checkpoint-0150 \
    outputs/olmoe-arith-d2d3-experts-s2/checkpoint-0150 \
  --output outputs/olmoe_d2d3_experts_150_d3_500_eval.json \
  2>&1 | tee outputs/olmoe_d2d3_experts_150_d3_500_eval.log

echo "########## ALL CONTROLS DONE $(date -Is) ##########"
python - <<'PY'
import json
from pathlib import Path
out = Path("outputs")
for name in [
    "olmoe_d2d3_attn_r128_d3_500_eval.json",
    "olmoe_d2d3_experts_d3_500_eval.json",
    "olmoe_d2d3_attn_rank128_d3_500_eval.json",
    "olmoe_d2d3_attn_lr3e5_d3_500_eval.json",
    "olmoe_d2d3_attn_r128_150_d3_500_eval.json",
    "olmoe_d2d3_experts_150_d3_500_eval.json",
]:
    path = out / name
    if not path.exists():
        print(name, "MISSING")
        continue
    data = json.loads(path.read_text())
    vals = [row["evaluation_success_fraction"] for row in data["checkpoints"]]
    print(name, vals, "mean=", data.get("mean_success_fraction"))
PY
