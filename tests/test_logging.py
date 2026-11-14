from grpx.logging import format_record


def test_format_record_separates_fields_with_pipe():
    text = format_record({
        "iteration": 1,
        "synchronize_seconds": 0.349,
        "peak_memory_gigabytes": 9.884,
    })
    assert "synchronize_seconds=0.349 | peak_memory_gigabytes=9.884" in text
    assert "0.349peak_memory" not in text
