"""采样后端。

HuggingFaceBackend 是默认本地路径，导入本包即可用。
VllmBackend 依赖 requests 等额外包，用到时再导入，避免本地后端被连带拖垮。
"""

from grpx.backends.huggingface import HuggingFaceBackend

__all__ = ["HuggingFaceBackend", "VllmBackend"]


def __getattr__(name):
    if name == "VllmBackend":
        from grpx.backends.vllm import VllmBackend
        return VllmBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
