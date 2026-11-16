"""pytest 钩子：network 标记的测试默认跳过，需显式 -m network 才跑。"""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "network: needs network / Hugging Face download; skipped by default",
    )


def pytest_collection_modifyitems(config, items):
    markexpr = (config.option.markexpr or "").strip()
    # 调用方显式选了 network（如 -m network）时不再跳过
    if "network" in markexpr:
        return
    import pytest

    skip_network = pytest.mark.skip(
        reason="network tests skipped by default; run with: pytest -m network")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)
