import pytest

from src.runner.logger import REQUIRED_RAW_RESULT_FIELDS, RawCSVLogger
from src.runner.result_row import _bytes_to_mb, build_raw_result_row
from src.workloads.uniform import make_uniform_workload
from tests.conftest import make_contiguous_cache, make_kv, make_paged_cache


TEST_HARDWARE_FIELDS = {
    "gpu_name": "cpu",
    "cuda_version": "none",
    "pytorch_version": "test",
    "driver_version": "none",
    "commit_hash": "abc123",
}


def test_bytes_to_mb():
    assert _bytes_to_mb(0) == pytest.approx(0.0)
    assert _bytes_to_mb(1024 * 1024) == pytest.approx(1.0)


def test_bytes_to_mb_rejects_negative_values():
    with pytest.raises(ValueError):
        _bytes_to_mb(-1)


def test_build_raw_result_row_for_contiguous_cache_has_required_schema():
    cache = make_contiguous_cache()
    workload = make_uniform_workload(batch_size=1, prompt_len=6, decode_len=2)

    for token_pos in range(6):
        key, value = make_kv(token_pos, token_pos)
        cache.write(layer_id=0, seq_id=0, token_pos=token_pos, key=key, value=value)

    row = build_raw_result_row(
        experiment_id="h1_uniform",
        cache=cache,
        workload=workload,
        cache_type="contiguous",
        block_size=0,
        workload_type="uniform",
        decode_len=2,
        run_id=0,
        warmup_runs=3,
        measured_runs=20,
        tpot_ms=1.0,
        throughput_tok_s=100.0,
        p50_latency_ms=1.0,
        p95_latency_ms=1.2,
        peak_gpu_memory_mb=0.0,
        oom_status=False,
        slow_status=False,
        hardware_fields=TEST_HARDWARE_FIELDS,
    )

    assert list(row.keys()) == REQUIRED_RAW_RESULT_FIELDS

    assert row["experiment_id"] == "h1_uniform"
    assert row["cache_type"] == "contiguous"
    assert row["block_size"] == 0
    assert row["batch_size"] == 1
    assert row["seq_len_min"] == 8
    assert row["seq_len_mean"] == pytest.approx(8.0)
    assert row["seq_len_max"] == 8
    assert row["used_tokens"] == 6
    assert row["wasted_tokens"] == 2
    assert row["fragmentation_ratio"] == pytest.approx(0.25)
    assert row["copy_bytes_per_token"] == 0.0
    assert row["page_lookups_per_token"] == 0.0
    assert row["commit_hash"] == "abc123"


def test_build_raw_result_row_for_paged_cache_tracks_copy_and_page_lookup():
    cache = make_paged_cache(block_size=4)
    workload = make_uniform_workload(batch_size=1, prompt_len=6, decode_len=2)

    for token_pos in range(6):
        key, value = make_kv(token_pos, token_pos)
        cache.write(layer_id=0, seq_id=0, token_pos=token_pos, key=key, value=value)

    cache.read(layer_id=0, seq_id=0, upto_pos=6)

    row = build_raw_result_row(
        experiment_id="h1_uniform",
        cache=cache,
        workload=workload,
        cache_type="paged_materialized",
        block_size=4,
        workload_type="uniform",
        decode_len=2,
        run_id=0,
        warmup_runs=3,
        measured_runs=20,
        tpot_ms=1.5,
        throughput_tok_s=80.0,
        p50_latency_ms=1.4,
        p95_latency_ms=1.8,
        peak_gpu_memory_mb=0.0,
        oom_status=False,
        slow_status=False,
        hardware_fields=TEST_HARDWARE_FIELDS,
    )

    assert list(row.keys()) == REQUIRED_RAW_RESULT_FIELDS

    assert row["cache_type"] == "paged_materialized"
    assert row["block_size"] == 4
    assert row["used_tokens"] == 6
    assert row["wasted_tokens"] == 2
    assert row["fragmentation_ratio"] == pytest.approx(0.25)
    assert row["copy_bytes_per_token"] > 0
    assert row["page_lookups_per_token"] > 0


def test_result_row_can_be_written_by_raw_csv_logger(tmp_path):
    cache = make_contiguous_cache()
    workload = make_uniform_workload(batch_size=1, prompt_len=6, decode_len=2)

    for token_pos in range(6):
        key, value = make_kv(token_pos, token_pos)
        cache.write(layer_id=0, seq_id=0, token_pos=token_pos, key=key, value=value)

    row = build_raw_result_row(
        experiment_id="h1_uniform",
        cache=cache,
        workload=workload,
        cache_type="contiguous",
        block_size=0,
        workload_type="uniform",
        decode_len=2,
        run_id=0,
        warmup_runs=3,
        measured_runs=20,
        tpot_ms=1.0,
        throughput_tok_s=100.0,
        p50_latency_ms=1.0,
        p95_latency_ms=1.2,
        peak_gpu_memory_mb=0.0,
        oom_status=False,
        slow_status=False,
        hardware_fields=TEST_HARDWARE_FIELDS,
    )

    output_path = tmp_path / "raw" / "result.csv"
    logger = RawCSVLogger(output_path)
    logger.write_rows([row])

    assert output_path.exists()


def test_build_raw_result_row_rejects_invalid_inputs():
    cache = make_contiguous_cache()
    workload = make_uniform_workload(batch_size=1, prompt_len=6, decode_len=2)

    with pytest.raises(ValueError):
        build_raw_result_row(
            experiment_id="",
            cache=cache,
            workload=workload,
            cache_type="contiguous",
            block_size=0,
            workload_type="uniform",
            decode_len=2,
            run_id=0,
            warmup_runs=3,
            measured_runs=20,
            tpot_ms=1.0,
            throughput_tok_s=100.0,
            p50_latency_ms=1.0,
            p95_latency_ms=1.2,
            peak_gpu_memory_mb=0.0,
            oom_status=False,
            slow_status=False,
            hardware_fields=TEST_HARDWARE_FIELDS,
        )
