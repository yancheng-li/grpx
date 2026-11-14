import torch
from transformers import GenerationConfig

from grpx.backends.common import normalize_stop_strings, parse_generated_tokens, resolve_end_token_ids

class HuggingFaceBackend:
    def __init__(self, model, tokenizer, max_new_tokens=32, temperature=1.0,
                 max_rows_per_batch=64, max_tokens_per_batch=32768, stop_strings=None):
        if not temperature > 0:
            raise ValueError(f"temperature 必须大于 0（采样分布为 softmax(logits / T)），收到 {temperature}")
        if max_rows_per_batch < 1:
            raise ValueError(f"max_rows_per_batch 至少为 1，收到 {max_rows_per_batch}")
        self.model, self.tokenizer = model, tokenizer
        self.max_new_tokens, self.temperature = max_new_tokens, temperature
        self.max_rows_per_batch, self.max_tokens_per_batch = max_rows_per_batch, max_tokens_per_batch
        self.stop_strings = normalize_stop_strings(stop_strings)
        self.end_token_ids = resolve_end_token_ids(model, tokenizer)

    def synchronize_policy(self, model):
        return

    def _shared_options(self):
        padding_token_id = self.tokenizer.pad_token_id
        if padding_token_id is None:
            padding_token_id = min(self.end_token_ids)
        return dict(repetition_penalty=1.0, max_new_tokens=self.max_new_tokens,
                    pad_token_id=padding_token_id, eos_token_id=sorted(self.end_token_ids))

    def _generation_config(self, greedy):
        if greedy:
            return GenerationConfig(do_sample=False, num_return_sequences=1,
                                    **self._shared_options())
        return GenerationConfig(do_sample=True, temperature=self.temperature, top_p=1.0,
                                top_k=0, num_return_sequences=1, **self._shared_options())

    def _split_into_chunks(self, ordered_indices, prompts):
        """ordered_indices 已按提示词长度升序，所以当前这条就是批内最长的"""
        chunks, current = [], []
        for index in ordered_indices:
            row_count = len(current) + 1
            padded_length = len(prompts[index]) + self.max_new_tokens
            if current and (row_count > self.max_rows_per_batch
                            or row_count * padded_length > self.max_tokens_per_batch):
                chunks.append(current)
                current = []
            current.append(index)
        if current:
            chunks.append(current)
        return chunks

    def _run_generation_batch(self, prompts, generation_config):
        """多条提示词（长度可不同），每条生成一个样本"""
        row_count = len(prompts)
        max_length = max(len(prompt) for prompt in prompts)
        input_token_ids = torch.full((row_count, max_length), generation_config.pad_token_id,
                                     dtype=torch.long)
        attention_mask = torch.zeros((row_count, max_length), dtype=torch.long)
        for row_index, prompt in enumerate(prompts):
            # 左侧填充：生成从序列末尾往后接，真实内容必须紧贴右边，填充由注意力掩码屏蔽
            start = max_length - len(prompt)
            input_token_ids[row_index, start:] = torch.tensor(prompt)
            attention_mask[row_index, start:] = 1
        input_token_ids = input_token_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)

        # 训练和采样共用模型对象；train 模式下 dropout（含 LoRA dropout）会扰动采样策略
        was_training = self.model.training
        self.model.eval()
        generate_options = dict(
            attention_mask=attention_mask,
            generation_config=generation_config,
        )
        if self.stop_strings:
            # 本地后端与 vLLM 对齐：生成到停止串就停；各行可停在不同位置
            generate_options["stop_strings"] = list(self.stop_strings)
            generate_options["tokenizer"] = self.tokenizer
        try:
            # 显式传入 generation_config，不继承模型自带默认，保证按策略分布原样采样
            output_token_ids = self.model.generate(input_token_ids, **generate_options)
        finally:
            self.model.train(was_training)

        results = []
        pad_token_id = generation_config.pad_token_id
        for generated_token_ids in output_token_ids[:, max_length:].tolist():
            if not self.stop_strings:
                # 停止串为空时保持旧行为与返回格式，同批对照不变
                end_position = next(
                    (position for position, token_id in enumerate(generated_token_ids)
                     if token_id in self.end_token_ids), None)
                if end_position is None:
                    results.append((generated_token_ids, True))
                else:
                    results.append((generated_token_ids[:end_position + 1], False))
                continue
            action_token_ids, finish_reason = parse_generated_tokens(
                generated_token_ids,
                self.end_token_ids,
                stop_strings=self.stop_strings,
                tokenizer=self.tokenizer,
                pad_token_id=pad_token_id,
                max_new_tokens=self.max_new_tokens,
            )
            results.append((
                action_token_ids, finish_reason == "length_limit", None, finish_reason))
        return results

    @torch.no_grad()
    def generate_batch(self, requests, greedy=False):
        """requests: [(提示词 token, 样本数[, url_index]), ...]；返回与 requests 等长的列表，
        每项是该请求的若干个（回答 token, 是否截断）。本地后端忽略 url_index。"""
        prompts, request_indices = [], []
        for request_index, request in enumerate(requests):
            prompt_token_ids, number_of_samples = request[0], request[1]
            prompts.extend([prompt_token_ids] * number_of_samples)
            request_indices.extend([request_index] * number_of_samples)

        ordered_indices = sorted(range(len(prompts)), key=lambda index: len(prompts[index]))
        generation_config = self._generation_config(greedy)
        outputs = [None] * len(prompts)
        for chunk in self._split_into_chunks(ordered_indices, prompts):
            chunk_outputs = self._run_generation_batch(
                [prompts[index] for index in chunk], generation_config)
            for index, output in zip(chunk, chunk_outputs, strict=True):
                outputs[index] = output

        results = [[] for _ in requests]
        for request_index, output in zip(request_indices, outputs, strict=True):
            results[request_index].append(output)
        return results

    def generate(self, prompt_token_ids, number_of_samples):
        return self.generate_batch([(prompt_token_ids, number_of_samples)])[0]

    def generate_greedy(self, prompt_token_ids):
        return self.generate_batch([(prompt_token_ids, 1)], greedy=True)[0][0]
