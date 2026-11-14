from typing import Protocol


class GenerationBackend(Protocol):
    temperature: float    # 写进 Trajectory.sampling_temperature，训练端按它缩放 logits

    def generate_batch(self, requests: list[tuple[list[int], int]], greedy: bool = False
                       ) -> list[list[tuple[list[int], bool]]]:
        """requests 是若干 (提示词 token, 样本数)；返回与 requests 等长的列表，
        每项是该请求的若干个 (生成的 token, 是否因长度上限被截断)。"""
        ...

    def generate(self, prompt_token_ids: list[int], number_of_samples: int) -> list[tuple[list[int], bool]]:
        """返回 number_of_samples 个 (生成的 token, 是否因长度上限被截断)。"""
        ...

    def generate_greedy(self, prompt_token_ids: list[int]) -> tuple[list[int], bool]:
        """贪心解码一条，返回 (生成的 token, 是否因长度上限被截断)。"""
        ...

    def synchronize_policy(self, model) -> None:
        """把训练端权重同步到采样后端；本地后端与训练共用对象，实现可为空。"""
        ...
