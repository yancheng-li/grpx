GENERATION_FINISH_REASONS = ("end_token", "stop_string", "length_limit")


def resolve_end_token_ids(model, tokenizer):
    end_token_ids = model.generation_config.eos_token_id
    if end_token_ids is None:
        end_token_ids = tokenizer.eos_token_id
    if end_token_ids is None:
        raise ValueError("model.generation_config 和 tokenizer 都没有 eos_token_id")
    if not isinstance(end_token_ids, list):
        end_token_ids = [end_token_ids]
    return set(end_token_ids)


def normalize_stop_strings(stop_strings):
    """配置里的停止串：None / 空 → ()；否则必须是非空字符串列表。"""
    if stop_strings is None:
        return ()
    if isinstance(stop_strings, str):
        raise ValueError("stop_strings 应是字符串列表，例如 [\"</calc>\"]，不能是单个字符串")
    strings = tuple(stop_strings)
    if any(not isinstance(item, str) or item == "" for item in strings):
        raise ValueError(f"stop_strings 的每一项都必须是非空字符串，收到 {stop_strings!r}")
    return strings


def parse_generated_tokens(token_ids, end_token_ids, stop_strings=(), tokenizer=None,
                           pad_token_id=None, max_new_tokens=None):
    """从生成 token 里切出动作，并判定结束原因。

    按从左到右扫描：先碰到结束 token → end_token（含该 token）；
    先碰到「解码文本已包含某个停止串」→ stop_string（含使停止串完整出现的那个 token，
    即使它后半截超出停止串）；否则视为 length_limit。
    动作文本应由返回的 token 解码得到，不要另用后端截断过的字符串。
    """
    ids = list(token_ids)
    if pad_token_id is not None:
        while ids and ids[-1] == pad_token_id and ids[-1] not in end_token_ids:
            ids.pop()
    stop_strings = tuple(stop_strings or ())
    if stop_strings and tokenizer is None:
        raise ValueError("配置了 stop_strings 时必须传入 tokenizer，才能按解码文本对齐 token 边界")

    for index, token_id in enumerate(ids):
        prefix = ids[: index + 1]
        if token_id in end_token_ids:
            return prefix, "end_token"
        if stop_strings:
            text = tokenizer.decode(prefix, skip_special_tokens=False)
            if any(stop in text for stop in stop_strings):
                return prefix, "stop_string"

    if max_new_tokens is not None and len(ids) >= max_new_tokens:
        return ids, "length_limit"
    return ids, "length_limit"


def is_lora_model(model):
    try:
        from peft import PeftModel
    except ImportError:
        return False
    return isinstance(model, PeftModel)
