#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/projects/grpx
source /root/miniconda3/etc/profile.d/conda.sh
conda activate verl-vllm
pkill -f 'vllm serve' 2>/dev/null || true
sleep 2
mkdir -p outputs
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
nohup vllm serve allenai/OLMoE-1B-7B-0924-Instruct \
  --enable-lora --max-lora-rank 32 \
  --generation-config vllm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 1024 \
  --port 8000 \
  > outputs/vllm_serve_experts_eval.log 2>&1 &
echo PID=$!
for i in $(seq 1 90); do
  if curl -sf http://localhost:8000/v1/models >/dev/null; then
    echo "READY after ${i}s"
    curl -s http://localhost:8000/v1/models | head -c 400
    echo
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
    exit 0
  fi
  if ! pgrep -af 'vllm serve' >/dev/null; then
    echo "vllm died"
    tail -n 60 outputs/vllm_serve_experts_eval.log
    exit 1
  fi
  sleep 2
done
echo TIMEOUT
tail -n 60 outputs/vllm_serve_experts_eval.log
exit 1
