import pytest

from src.metrics.latency import (
    measure_elapsed_ms,
    percentile,
    summarize_latencies_ms,
    synchronize_if_needed,
    throughput_tokens_per_second,
    time_per_output_token_ms,
)


def test_synchronize_if_needed_cpu_does_not_crash():
    synchronize_if_needed("cpu")


def test_measure_elapsed_ms_returns_result_and_nonnegative_time():
    def fn():
        return sum(range(10))

    result, elapsed_ms = measure_elapsed_ms(fn, device="cpu")

    assert result == 45
    assert elapsed_ms >= 0.0


def test_percentile_single_value():
    assert percentile([5.0], 50) == 5.0


def test_percentile_linear_interpolation():
    values = [1.0, 2.0, 3.0, 4.0]

    assert percentile(values, 0) == pytest.approx(1.0)
    assert percentile(values, 50) == pytest.approx(2.5)
    assert percentile(values, 100) == pytest.approx(4.0)


def test_percentile_rejects_empty_values():
    with pytest.raises(ValueError):
        percentile([], 50)


def test_percentile_rejects_invalid_percentile():
    with pytest.raises(ValueError):
        percentile([1.0, 2.0], -1)

    with pytest.raises(ValueError):
        percentile([1.0, 2.0], 101)


def test_summarize_latencies_ms():
    summary = summarize_latencies_ms([1.0, 2.0, 3.0, 4.0])

    assert summary["count"] == 4
    assert summary["mean_ms"] == pytest.approx(2.5)
    assert summary["p50_ms"] == pytest.approx(2.5)
    assert summary["p95_ms"] == pytest.approx(3.85)
    assert summary["min_ms"] == pytest.approx(1.0)
    assert summary["max_ms"] == pytest.approx(4.0)


def test_summarize_latencies_rejects_empty_input():
    with pytest.raises(ValueError):
        summarize_latencies_ms([])


def test_summarize_latencies_rejects_negative_values():
    with pytest.raises(ValueError):
        summarize_latencies_ms([1.0, -1.0])


def test_time_per_output_token_ms():
    assert time_per_output_token_ms(
        total_elapsed_ms=100.0,
        num_output_tokens=20,
    ) == pytest.approx(5.0)


def test_time_per_output_token_ms_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        time_per_output_token_ms(total_elapsed_ms=-1.0, num_output_tokens=20)

    with pytest.raises(ValueError):
        time_per_output_token_ms(total_elapsed_ms=100.0, num_output_tokens=0)


def test_throughput_tokens_per_second():
    assert throughput_tokens_per_second(
        num_tokens=100,
        total_elapsed_ms=1000.0,
    ) == pytest.approx(100.0)


def test_throughput_tokens_per_second_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        throughput_tokens_per_second(num_tokens=0, total_elapsed_ms=1000.0)

    with pytest.raises(ValueError):
        throughput_tokens_per_second(num_tokens=100, total_elapsed_ms=0.0)
