import concurrent.futures
import json
import pathlib
import re
import shutil
import threading
import time

import requests

from grpx.backends.common import is_lora_model

RETRYABLE_STATUS_CODES = {500, 502, 503, 504}
MAX_ATTEMPTS = 3


def base_model_ids_from_vllm_models(payload):
    """从 /v1/models 响应里取出基座模型 id（parent 为 null 的项）。"""
    if not isinstance(payload, dict):
        return []
    return [
        item["id"] for item in payload.get("data", [])
        if isinstance(item, dict) and item.get("parent") is None and "id" in item
    ]


def verify_vllm_base_model_match(payload, expected_name):
    """服务端基座必须包含 expected_name，否则抛错。"""
    bases = base_model_ids_from_vllm_models(payload)
    if expected_name in bases:
        return bases
    listed = "、".join(bases) if bases else "（无）"
    raise RuntimeError(
        f"训练端和推理服务不是同一个模型：期望基座 {expected_name!r}，"
        f"服务端基座为 {listed}。"
        "请用同一模型启动 vLLM，或在 [backend] 设置 server_model_name 声明服务端用的名字"
        "（别名 / 本地路径）。"
    )


def sanitize_adapter_name_component(text):
    """适配器名只保留字母数字、点、下划线和短横线，避免运行名带空格等非法字符。"""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-._")
    return cleaned or "run"


def normalize_base_urls(base_url):
    """字符串或列表 → 去尾斜杠后的非空地址列表。"""
    if isinstance(base_url, str):
        urls = [base_url]
    elif isinstance(base_url, (list, tuple)):
        urls = list(base_url)
    else:
        raise ValueError(
            f"base_url 必须是字符串或字符串列表，收到 {type(base_url).__name__}")
    if not urls:
        raise ValueError("base_url 不能为空")
    normalized = []
    for item in urls:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("base_url 列表里每一项都必须是非空字符串")
        normalized.append(item.rstrip("/"))
    return normalized


def _normalize_path_for_prefix(path):
    """统一分隔符，并把盘符收成小写，避免 D:/ 配成 d:/ 时匹配失败。"""
    text = str(path).replace("\\", "/")
    if len(text) >= 2 and text[1] == ":":
        text = text[0].lower() + text[1:]
    return text


def _prefix_covers(path, prefix):
    """前缀匹配，且落在路径边界上：D:/projects 不匹配 D:/projects2。"""
    normalized_path = _normalize_path_for_prefix(path)
    normalized_prefix = _normalize_path_for_prefix(prefix).rstrip("/")
    if normalized_path == normalized_prefix:
        return True
    return normalized_path.startswith(normalized_prefix + "/")


class VllmBackend:
    """远程 vLLM 采样后端。

    同步用「前缀-版本号」适配器名（如 policy-<运行名>-1），避免同名原地替换命中旧前缀缓存。
    调用方应把运行名编进 adapter_name，保证跨运行也不撞名。

    首次同步前会卸掉服务端同前缀的残留版本（resume 时版本号从 1 重计，否则会撞名）。
    若仍遇到 already been loaded，自动 unload 后再 load 一次。
    close() 会卸掉当前适配器，减少进程退出后的残留。

    重新同步只覆盖「服务仍在、但适配器被清掉」：连接失败（服务重启）不会在这里
    长时间等待，需要先把服务拉起来再跑 doctor / 训练。
    """

    def __init__(self, tokenizer, end_token_ids, adapter_directory,
                 base_url="http://localhost:8000", adapter_name="policy",
                 max_new_tokens=32, temperature=1.0,
                 max_concurrent_requests=64, request_timeout_seconds=600,
                 adapter_path_local_prefix=None, adapter_path_server_prefix=None,
                 expected_base_model=None, training_model_name=None,
                 stop_strings=None):
        from grpx.backends.common import normalize_stop_strings

        if not temperature > 0:
            raise ValueError(f"temperature 必须大于 0（采样分布为 softmax(logits / T)），收到 {temperature}")
        if max_concurrent_requests < 1:
            raise ValueError(f"max_concurrent_requests 至少为 1，收到 {max_concurrent_requests}")
        if (adapter_path_local_prefix is None) != (adapter_path_server_prefix is None):
            raise ValueError("adapter_path_local_prefix 与 adapter_path_server_prefix 必须同时配置或同时省略")
        if not expected_base_model:
            raise ValueError("expected_base_model 不能为空（应与 [model].name 或 server_model_name 一致）")
        self.tokenizer = tokenizer
        self.end_token_ids = set(end_token_ids)
        self.default_end_token_id = tokenizer.eos_token_id
        if self.default_end_token_id is None:
            self.default_end_token_id = min(self.end_token_ids)
        self.adapter_directory = pathlib.Path(adapter_directory)
        self.base_urls = normalize_base_urls(base_url)
        self.base_url = self.base_urls[0]  # 单地址旧代码 / 日志兼容
        self.adapter_name_prefix = sanitize_adapter_name_component(adapter_name)
        self.adapter_name = None    # 当前在用的带版本号名字；首次同步前为 None
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.stop_strings = normalize_stop_strings(stop_strings)
        self.request_timeout_seconds = request_timeout_seconds
        self.adapter_path_local_prefix = adapter_path_local_prefix
        self.adapter_path_server_prefix = adapter_path_server_prefix
        self.expected_base_model = expected_base_model
        # [model].name；与 expected（常为 server_model_name）不同时用于放行本地路径
        self.training_model_name = training_model_name or expected_base_model
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_requests)
        self._policy_model = None
        self._synchronization_lock = threading.Lock()
        self._synchronization_count = 0
        self._adapter_version = 0
        self._active_adapter_directory = None
        self._cleared_stale_adapters = False
        self._verified_server_base = False
        # 只有 routing_replay 打开时才会在采样响应里读取专家记录。贪心评测不读。
        self.record_routing = False
        self.routing_layer_count = None
        self.routing_experts_per_token = None

    @property
    def base_url_count(self):
        return len(self.base_urls)

    def close(self):
        if self.adapter_name is not None:
            for url in self.base_urls:
                self._unload_adapter(self.adapter_name, raise_on_error=False, base_url=url)
            self.adapter_name = None
            self._active_adapter_directory = None
        self.executor.shutdown(wait=False, cancel_futures=True)

    def server_adapter_path(self, directory=None):
        target = directory if directory is not None else self._active_adapter_directory
        if target is None:
            target = self.adapter_directory
        absolute = str(pathlib.Path(target).resolve())
        local_prefix = self.adapter_path_local_prefix
        server_prefix = self.adapter_path_server_prefix
        if local_prefix is None:
            return absolute
        if not _prefix_covers(absolute, local_prefix):
            raise ValueError(
                f"适配器目录 {absolute} 不在 adapter_path_local_prefix={local_prefix!r} 之下；"
                "请把运行目录放在该前缀内，或改正前缀配置")
        normalized_absolute = _normalize_path_for_prefix(absolute)
        normalized_local = _normalize_path_for_prefix(local_prefix).rstrip("/")
        suffix = normalized_absolute[len(normalized_local):]
        return server_prefix.rstrip("/") + suffix

    def synchronize_policy(self, model):
        if not is_lora_model(model):
            raise ValueError(
                "vLLM 后端需要 LoRA 模型（synchronize_policy 靠保存适配器同步权重）；"
                "非 LoRA 请改用 backend.type = \"huggingface\""
            )
        with self._synchronization_lock:
            self._synchronize_locked(model)

    def _synchronize_locked(self, model):
        """调用方必须已持有 _synchronization_lock。

        适配器只存一份，对每个 base_url 加载；全部成功才切换当前版本。
        任一地址失败：卸掉已装上的新适配器并报错，避免副本版本分裂。
        切换成功后卸载旧版本：某个地址失败只警告（新版本已全部就位）。
        """
        if not self._cleared_stale_adapters:
            self._clear_stale_prefix_adapters()
            self._cleared_stale_adapters = True
        self._policy_model = model
        self._adapter_version += 1
        new_name = f"{self.adapter_name_prefix}-{self._adapter_version}"
        new_directory = self.adapter_directory / new_name
        if new_directory.exists():
            shutil.rmtree(new_directory)
        new_directory.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(new_directory)

        old_name = self.adapter_name
        old_directory = self._active_adapter_directory
        loaded_urls = []
        try:
            for url in self.base_urls:
                try:
                    self._load_adapter(new_name, new_directory, base_url=url)
                except RuntimeError as error:
                    if (old_name is not None and self._is_lora_capacity_error(error)
                            and len(self.base_urls) == 1):
                        # 单地址槽位满：先卸旧的再加载新的
                        self._unload_adapter(old_name, raise_on_error=True, base_url=url)
                        self._remove_adapter_directory(old_directory)
                        old_name, old_directory = None, None
                        self._load_adapter(new_name, new_directory, base_url=url)
                    elif old_name is not None and self._is_lora_capacity_error(error):
                        # 多地址：各卸本地址旧槽后再装；仍失败则整体回滚
                        self._unload_adapter(old_name, raise_on_error=True, base_url=url)
                        self._load_adapter(new_name, new_directory, base_url=url)
                    else:
                        raise
                loaded_urls.append(url)
        except Exception:
            for url in loaded_urls:
                self._unload_adapter(new_name, raise_on_error=False, base_url=url)
            self._remove_adapter_directory(new_directory)
            raise

        self.adapter_name = new_name
        self._active_adapter_directory = new_directory
        self._synchronization_count += 1

        if old_name is not None:
            for url in self.base_urls:
                self._unload_adapter(old_name, raise_on_error=False, base_url=url)
            self._remove_adapter_directory(old_directory)

    def _is_versioned_adapter_name(self, name):
        """仅匹配「前缀-数字」，避免 policy-run 误伤 policy-run-extra-1。"""
        return bool(re.fullmatch(rf"{re.escape(self.adapter_name_prefix)}-\d+", name))

    def _list_server_models_payload(self, base_url=None):
        url = (base_url or self.base_url).rstrip("/")
        response = self._request(
            "GET", f"{url}/v1/models", None, allow_resync=False)
        payload = response.json()
        return payload if isinstance(payload, dict) else {"data": []}

    def _list_server_model_ids(self, base_url=None):
        payload = self._list_server_models_payload(base_url=base_url)
        return [item["id"] for item in payload.get("data", [])
                if isinstance(item, dict) and "id" in item]

    def verify_server_base_model(self):
        """对每个 base_url 请求 /v1/models，确认基座与 expected_base_model 一致。"""
        last_payload = None
        for url in self.base_urls:
            payload = self._list_server_models_payload(base_url=url)
            try:
                verify_vllm_base_model_match(payload, self.expected_base_model)
            except RuntimeError as error:
                raise RuntimeError(f"{url}：{error}") from error
            last_payload = payload
        self._verified_server_base = True
        return last_payload

    def _clear_stale_prefix_adapters(self):
        """卸掉各服务端同前缀的残留版本（典型：上一进程退出后留下的最后一个）。"""
        if not self._verified_server_base:
            self.verify_server_base_model()
        for url in self.base_urls:
            try:
                payload = self._list_server_models_payload(base_url=url)
                model_ids = [item["id"] for item in payload.get("data", [])
                             if isinstance(item, dict) and "id" in item]
            except RuntimeError as error:
                if "不是同一个模型" in str(error):
                    raise
                print(f"列出服务端适配器失败（{url}，忽略，继续同步）：{error}", flush=True)
                continue
            for name in model_ids:
                if not self._is_versioned_adapter_name(name):
                    continue
                print(f"卸掉残留适配器 {name} @ {url}", flush=True)
                self._unload_adapter(name, raise_on_error=False, base_url=url)

    def _adapter_parent_on_server(self, name, payload=None, base_url=None):
        if payload is None:
            payload = self._list_server_models_payload(base_url=base_url)
        for item in payload.get("data", []):
            if isinstance(item, dict) and item.get("id") == name:
                return item.get("parent")
        return None

    def _trained_base_model_name(self, directory):
        config_path = pathlib.Path(directory) / "adapter_config.json"
        if not config_path.exists():
            return None
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if isinstance(payload, dict):
            return payload.get("base_model_name_or_path")
        return None

    def _verify_loaded_adapter(self, name, directory, base_url=None):
        """vLLM 对基座不匹配的 LoRA 仍可能返回 HTTP 200；用 parent 与 adapter_config 核对。"""
        url = base_url or self.base_url
        payload = self._list_server_models_payload(base_url=url)
        parent = self._adapter_parent_on_server(name, payload, base_url=url)
        trained_base = self._trained_base_model_name(directory)
        if parent != self.expected_base_model:
            raise RuntimeError(
                f"{url}：适配器 {name} 挂在服务端基座 {parent!r} 上，"
                f"期望 {self.expected_base_model!r}；"
                "训练端和推理服务不是同一个模型"
            )
        # adapter_config 可能是本地路径，服务端 parent 是模型名：与第 1 项一样，
        # 用 training_model_name（[model].name）与 expected_base_model（server_model_name）放行
        allowed_trained = {parent, self.expected_base_model, self.training_model_name}
        if trained_base is not None and trained_base not in allowed_trained:
            raise RuntimeError(
                f"{url}：适配器基于 {trained_base!r}，但服务端基座是 {parent!r}；"
                "vLLM 的 /v1/load_lora_adapter 在形状不匹配时仍可能返回成功，"
                "采样实际走的是服务端基座。请确认 [model].name 与 vLLM 加载的是同一模型"
                "（本地路径与服务端名字不一致时，在 [backend] 设置 server_model_name）"
            )

    def _load_adapter(self, name, directory, base_url=None):
        url = (base_url or self.base_url).rstrip("/")
        payload = {
            "lora_name": name,
            "lora_path": self.server_adapter_path(directory),
        }
        try:
            self._request(
                "POST",
                f"{url}/v1/load_lora_adapter",
                payload,
                allow_resync=False,
            )
        except RuntimeError as error:
            if not self._is_already_loaded_error(error):
                raise
            # resume 或清残留失败时的兜底：卸掉同名再 load 一次
            print(f"适配器 {name} 已在服务端（{url}），先卸载再加载", flush=True)
            self._unload_adapter(name, raise_on_error=False, base_url=url)
            self._request(
                "POST",
                f"{url}/v1/load_lora_adapter",
                payload,
                allow_resync=False,
            )
        self._verify_loaded_adapter(name, directory, base_url=url)

    def _unload_adapter(self, name, raise_on_error, base_url=None):
        url = (base_url or self.base_url).rstrip("/")
        try:
            self._request(
                "POST",
                f"{url}/v1/unload_lora_adapter",
                {"lora_name": name},
                allow_resync=False,
            )
        except RuntimeError as error:
            if raise_on_error:
                raise
            print(f"卸载旧适配器 {name} @ {url} 失败（忽略，继续训练）：{error}", flush=True)

    @staticmethod
    def _remove_adapter_directory(directory):
        if directory is None:
            return
        shutil.rmtree(directory, ignore_errors=True)

    @staticmethod
    def _is_lora_capacity_error(error):
        message = str(error).lower()
        return "lora" in message and any(
            token in message for token in ("max", "capacity", "limit", "exceed", "槽")
        )

    @staticmethod
    def _is_already_loaded_error(error):
        message = str(error).lower()
        return "already been loaded" in message or "already loaded" in message

    def _resynchronize_if_stale(self, observed_count):
        with self._synchronization_lock:
            if self._synchronization_count == observed_count:
                self._synchronize_locked(self._policy_model)

    def _error_message(self, response):
        text = response.text or ""
        try:
            payload = response.json()
        except (ValueError, json.JSONDecodeError):
            return text
        if not isinstance(payload, dict):
            return text
        error = payload.get("error", payload)
        if isinstance(error, dict):
            return error.get("message") or text
        if isinstance(error, str):
            return error
        return text

    def _is_model_missing(self, response):
        if response is None or self.adapter_name is None:
            return False
        if response.status_code not in (404, 400):
            return False
        message = self._error_message(response).lower()
        return self.adapter_name.lower() in message and (
            "not found" in message or "does not exist" in message or "不存在" in message
        )

    def _request(self, method, url, payload, allow_resync=True):
        """超时与 5xx 会消耗尝试次数；因适配器丢失而重新同步不消耗次数。

        不处理 ConnectionError（服务重启）：那种情况需要人为把服务拉起来。
        """
        remaining_attempts = MAX_ATTEMPTS
        last_error = None
        while remaining_attempts > 0:
            observed_count = self._synchronization_count
            try:
                response = requests.request(
                    method, url, json=payload, timeout=self.request_timeout_seconds)
            except requests.Timeout as error:
                last_error = error
                remaining_attempts -= 1
                print(
                    f"请求重试：{url} 超时（剩余尝试 {remaining_attempts}）：{error}",
                    flush=True,
                )
                if remaining_attempts == 0:
                    raise RuntimeError(f"请求 {url} 超时：{error}") from error
                time.sleep(0.5 * (2 ** (MAX_ATTEMPTS - remaining_attempts - 1)))
                continue
            except requests.RequestException as error:
                # 含 ConnectionError：服务重启不在自动恢复范围内，见类文档字符串
                raise RuntimeError(f"请求 {url} 失败：{error}") from error

            if response.status_code < 400:
                return response

            detail = self._error_message(response)
            if (allow_resync and self._policy_model is not None
                    and self._is_model_missing(response)):
                print(
                    f"请求重试：{url} 适配器丢失，重新同步后重试：{detail}",
                    flush=True,
                )
                self._resynchronize_if_stale(observed_count)
                continue
            if response.status_code in RETRYABLE_STATUS_CODES:
                last_error = RuntimeError(f"HTTP {response.status_code}：{detail}")
                remaining_attempts -= 1
                print(
                    f"请求重试：{url} HTTP {response.status_code}"
                    f"（剩余尝试 {remaining_attempts}）：{detail}",
                    flush=True,
                )
                if remaining_attempts == 0:
                    raise last_error
                time.sleep(0.5 * (2 ** (MAX_ATTEMPTS - remaining_attempts - 1)))
                continue
            raise RuntimeError(f"HTTP {response.status_code}：{detail}")
        raise RuntimeError(f"请求 {url} 失败：{last_error}")

    def _parse_token_ids(self, choice):
        logprobs = choice.get("logprobs")
        if logprobs is None:
            return []
        tokens = logprobs.get("tokens")
        if not tokens:
            return []
        token_ids = []
        for token in tokens:
            try:
                prefix, _, value = token.partition(":")
                if prefix != "token_id" or not value:
                    raise ValueError
                token_ids.append(int(value))
            except ValueError:
                sample = tokens[:3]
                raise RuntimeError(
                    "服务返回的 token 格式不符合预期，请检查 vLLM 版本是否支持 "
                    f"return_tokens_as_token_ids；样例：{sample!r}"
                ) from None
        return token_ids

    def server_versions(self):
        """每个推理服务的版本号，写进运行配置，便于以后对上是哪个 vLLM。"""
        versions = []
        for url in self.base_urls:
            response = self._request("GET", f"{url}/version", None)
            payload = response.json()
            version = payload.get("version") if isinstance(payload, dict) else None
            if not version:
                raise RuntimeError(f"推理服务没有返回版本号：{url}")
            versions.append({"base_url": url, "version": str(version)})
        return versions

    def probe_routed_experts(self, model):
        """重放打开时，先打一条请求。响应里没有路由字段就停，不进入训练。"""
        from grpx.routing_replay import (
            _router_attribute,
            decode_vllm_routed_experts,
            find_router_modules,
        )

        modules = find_router_modules(model)
        if not modules:
            raise RuntimeError("routing_replay 只适用于带路由的模型")
        layer_count = len(modules)
        experts_per_token = int(_router_attribute(next(iter(modules.values())), "top_k"))
        self.routing_layer_count = layer_count
        self.routing_experts_per_token = experts_per_token
        prompt = [1, 2, 3, 4]
        for url in self.base_urls:
            response = self._request("POST", f"{url}/v1/completions", {
                "model": self.expected_base_model,
                "prompt": prompt,
                "n": 1,
                "max_tokens": 1,
                "temperature": 0.0,
                "top_p": 1.0,
                "logprobs": 1,
                "return_tokens_as_token_ids": True,
            })
            choice = response.json()["choices"][0]
            token_ids = self._parse_token_ids(choice)
            if not token_ids:
                raise RuntimeError(f"探测路由时推理服务没有返回 token：{url}")
            decode_vllm_routed_experts(
                choice.get("routed_experts"),
                len(prompt),
                len(token_ids),
                layer_count,
                experts_per_token,
            )

    def _complete(self, prompt_token_ids, number_of_samples, greedy, base_url=None):
        # vLLM 贪心采样要求 n=1；评测时多条轨迹常共享同一开场提示词，会被合并成 n>1。
        # 贪心结果确定，请求一次后复制即可。
        url = (base_url or self.base_url).rstrip("/")
        request_count = 1 if greedy else number_of_samples
        payload = {
            "model": self.adapter_name,
            "prompt": prompt_token_ids,
            "n": request_count,
            "max_tokens": self.max_new_tokens,
            "temperature": 0.0 if greedy else self.temperature,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.0,
            "stop_token_ids": sorted(self.end_token_ids),
            "logprobs": 1,
            "return_tokens_as_token_ids": True,
            "skip_special_tokens": False,
        }
        if self.stop_strings:
            payload["stop"] = list(self.stop_strings)
            payload["include_stop_str_in_output"] = True
        response = self._request("POST", f"{url}/v1/completions", payload)
        results = []
        for choice in sorted(response.json()["choices"], key=lambda choice: choice["index"]):
            api_finish_reason = choice.get("finish_reason")
            if api_finish_reason not in ("stop", "length"):
                raise RuntimeError(
                    f"未预期的 finish_reason={api_finish_reason!r}；"
                    "只接受 'stop'（正常结束）或 'length'（截断）")
            token_ids = self._parse_token_ids(choice)
            logprobs = choice.get("logprobs") or {}
            token_logprobs = list(logprobs.get("token_logprobs") or [])
            stop_reason = choice.get("stop_reason")
            server_action_length = len(token_ids)
            if api_finish_reason == "length":
                finish_reason = "length_limit"
                was_truncated = True
            elif isinstance(stop_reason, str) or (
                    self.stop_strings and token_ids and any(
                        stop in self.tokenizer.decode(token_ids, skip_special_tokens=False)
                        for stop in self.stop_strings)):
                # 被停止串截住：不要补结束 token，训练算到停止串最后一个 token 为止
                finish_reason = "stop_string"
                was_truncated = False
            else:
                finish_reason = "end_token"
                was_truncated = False
                if not token_ids or token_ids[-1] not in self.end_token_ids:
                    # 服务器返回的 token 里不包含结束符时补上，训练端需要它来学习何时停止
                    token_ids.append(
                        stop_reason if isinstance(stop_reason, int) else self.default_end_token_id
                    )
            # 与 action_token_ids 对齐：补上的结束符没有服务端概率，记为 None 占位
            if len(token_logprobs) < len(token_ids):
                token_logprobs = token_logprobs + [None] * (len(token_ids) - len(token_logprobs))
            elif len(token_logprobs) > len(token_ids):
                token_logprobs = token_logprobs[:len(token_ids)]
            routed_experts = None
            if self.record_routing and not greedy:
                if self.routing_layer_count is None or self.routing_experts_per_token is None:
                    raise RuntimeError("还没有确认推理服务会返回路由")
                from grpx.routing_replay import decode_vllm_routed_experts
                routed_experts = decode_vllm_routed_experts(
                    choice.get("routed_experts"),
                    len(prompt_token_ids),
                    server_action_length,
                    self.routing_layer_count,
                    self.routing_experts_per_token,
                    extra_unrecorded=len(token_ids) - server_action_length,
                )
            if routed_experts is not None:
                results.append((
                    token_ids, was_truncated, token_logprobs, finish_reason, routed_experts))
            elif self.stop_strings:
                results.append((token_ids, was_truncated, token_logprobs, finish_reason))
            else:
                # 停止串为空时保持旧三元组，同批对照不变
                results.append((token_ids, was_truncated, token_logprobs))
        if greedy and number_of_samples > 1:
            assert len(results) == 1
            results = results * number_of_samples
        return results

    def generate_batch(self, generation_requests, greedy=False):
        """requests: [(提示词, 样本数[, url_index]), ...]；url_index 默认 0。"""
        futures = []
        for request in generation_requests:
            prompt_token_ids, number_of_samples = request[0], request[1]
            url_index = request[2] if len(request) > 2 else 0
            if url_index < 0 or url_index >= len(self.base_urls):
                raise ValueError(
                    f"url_index={url_index} 超出 base_urls 范围（共 {len(self.base_urls)} 个）")
            futures.append(self.executor.submit(
                self._complete,
                prompt_token_ids,
                number_of_samples,
                greedy,
                self.base_urls[url_index],
            ))
        results = []
        try:
            for future in futures:
                results.append(future.result())
        except BaseException:
            for future in futures:
                future.cancel()
            raise
        return results

    def generate(self, prompt_token_ids, number_of_samples):
        return [
            (token_ids, truncated)
            for token_ids, truncated, *_ in self.generate_batch(
                [(prompt_token_ids, number_of_samples)])[0]
        ]

    def generate_greedy(self, prompt_token_ids):
        token_ids, truncated, *_ = self.generate_batch(
            [(prompt_token_ids, 1)], greedy=True)[0][0]
        return token_ids, truncated
