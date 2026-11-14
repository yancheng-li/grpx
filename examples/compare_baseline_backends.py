"""未训练模型：HuggingFace vs vLLM 贪心评测交叉核对。

同一份 200 题、同一份随机初始化 LoRA（seed=0）。
期望本地后端：success=0.110，mean_reward=0.1545。
"""
import time
from pathlib import Path

from examples.guess_number import GuessNumberEnvironment, make_dataset
from grpx.backends.common import resolve_end_token_ids
from grpx.backends.huggingface import HuggingFaceBackend
from grpx.backends.vllm import VllmBackend
from grpx.command_line import load_model
from grpx.rollout import rollout_groups
from grpx.types import mean_environment_metrics, success_fraction

MODEL = {"name": "Qwen/Qwen2.5-0.5B-Instruct", "lora_rank": 16, "lora_alpha": 32}
SEED = 0
MAX_STEPS = 6
MAX_NEW_TOKENS = 32
EVALUATION_OPTIONS = {"size": 200, "seed": 1}
ENVIRONMENT_OPTIONS = {"reveal_interval": True}
EXPECTED = {"evaluation_success_fraction": 0.110, "evaluation_mean_reward": 0.1545}
VLLM_BASE_URL = "http://localhost:8000"


def metrics_from_trajectories(trajectories):
    count = len(trajectories)
    successes = success_fraction(trajectories)
    return {
        "evaluation_mean_reward":
            sum(trajectory.total_reward for trajectory in trajectories) / count,
        **({"evaluation_success_fraction": successes} if successes is not None else {}),
        "evaluation_length_limit_fraction":
            sum(trajectory.termination_reason == "length_limit" for trajectory in trajectories) / count,
        "evaluation_mean_step_count":
            sum(len(trajectory.steps) for trajectory in trajectories) / count,
        **mean_environment_metrics(trajectories, prefix="evaluation_environment_"),
    }


def run_greedy(backend, tokenizer, dataset):
    return rollout_groups(
        backend, tokenizer,
        lambda: GuessNumberEnvironment(**ENVIRONMENT_OPTIONS),
        [(f"evaluation-{index}", sample, index) for index, sample in enumerate(dataset)],
        group_size=1, max_steps=MAX_STEPS, greedy=True,
    )


def summarize(label, metrics, seconds):
    success = metrics["evaluation_success_fraction"]
    reward = metrics["evaluation_mean_reward"]
    print(f"{label}:")
    print(f"  success={success:.4f}  mean_reward={reward:.4f}  "
          f"steps={metrics['evaluation_mean_step_count']:.3f}  "
          f"length_limit={metrics['evaluation_length_limit_fraction']:.3f}  "
          f"seconds={seconds:.1f}")
    for key, value in metrics.items():
        if key.startswith("evaluation_environment_"):
            print(f"  {key.removeprefix('evaluation_')}: {value}")
    return success, reward


def main():
    dataset = make_dataset(**EVALUATION_OPTIONS)
    print(f"评测集 {len(dataset)} 题（seed={EVALUATION_OPTIONS['seed']}），"
          f"max_steps={MAX_STEPS}，reveal_interval={ENVIRONMENT_OPTIONS['reveal_interval']}")
    print(f"期望本地：success={EXPECTED['evaluation_success_fraction']:.3f}  "
          f"mean_reward={EXPECTED['evaluation_mean_reward']}")

    print("加载模型（float32 + 随机 LoRA，seed=0）……")
    model, tokenizer = load_model(MODEL, SEED)

    hf_backend = HuggingFaceBackend(model, tokenizer, max_new_tokens=MAX_NEW_TOKENS)
    started = time.perf_counter()
    hf_trajectories = run_greedy(hf_backend, tokenizer, dataset)
    hf_seconds = time.perf_counter() - started
    hf_metrics = metrics_from_trajectories(hf_trajectories)
    hf_success, hf_reward = summarize("HuggingFace", hf_metrics, hf_seconds)
    print(f"  vs 期望：success Δ={hf_success - EXPECTED['evaluation_success_fraction']:+.4f}  "
          f"reward Δ={hf_reward - EXPECTED['evaluation_mean_reward']:+.4f}")

    adapter_directory = Path("outputs/baseline-backend-compare/vllm_adapter")
    adapter_directory.mkdir(parents=True, exist_ok=True)
    vllm_backend = VllmBackend(
        tokenizer, resolve_end_token_ids(model, tokenizer),
        adapter_directory=adapter_directory,
        base_url=VLLM_BASE_URL,
        max_new_tokens=MAX_NEW_TOKENS,
        max_concurrent_requests=64,
    )
    print(f"同步 LoRA 到 vLLM（{adapter_directory.resolve()}）……")
    vllm_backend.synchronize_policy(model)

    started = time.perf_counter()
    vllm_trajectories = run_greedy(vllm_backend, tokenizer, dataset)
    vllm_seconds = time.perf_counter() - started
    vllm_metrics = metrics_from_trajectories(vllm_trajectories)
    vllm_success, vllm_reward = summarize("vLLM", vllm_metrics, vllm_seconds)
    print(f"  vs HuggingFace：success Δ={vllm_success - hf_success:+.4f}  "
          f"reward Δ={vllm_reward - hf_reward:+.4f}")

    mismatched_success = sum(
        left.is_successful != right.is_successful
        for left, right in zip(hf_trajectories, vllm_trajectories, strict=True)
    )
    mismatched_reward = sum(
        left.total_reward != right.total_reward
        for left, right in zip(hf_trajectories, vllm_trajectories, strict=True)
    )
    mismatched_actions = sum(
        [step.action_token_ids for step in left.steps]
        != [step.action_token_ids for step in right.steps]
        for left, right in zip(hf_trajectories, vllm_trajectories, strict=True)
    )
    print("逐题核对：")
    print(f"  成功标记不同：{mismatched_success}/{len(hf_trajectories)}")
    print(f"  回报不同：{mismatched_reward}/{len(hf_trajectories)}")
    print(f"  动作 token 序列不同：{mismatched_actions}/{len(hf_trajectories)}")

    vllm_backend.close()


if __name__ == "__main__":
    main()
