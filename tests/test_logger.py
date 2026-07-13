import csv

import pytest

from src.runner.logger import REQUIRED_RAW_RESULT_FIELDS, RawCSVLogger


def make_complete_row(**overrides):
    row = {
        "experiment_id": "h1_uniform",
        "cache_type": "contiguous",
        "block_size": 0,
        "workload_type": "uniform",
        "batch_size": 4,
        "seq_len_min": 128,
        "seq_len_mean": 128.0,
        "seq_len_max": 128,
        "decode_len": 60,
        "run_id": 0,
        "warmup_runs": 3,
        "measured_runs": 20,
        "tpot_ms": 1.23,
        "throughput_tok_s": 100.0,
        "p50_latency_ms": 1.20,
        "p95_latency_ms": 1.50,
        "peak_gpu_memory_mb": 0.0,
        "live_kv_memory_mb": 0.0,
        "allocated_pages": 0,
        "free_pages": 0,
        "used_tokens": 512,
        "wasted_tokens": 0,
        "fragmentation_ratio": 0.0,
        "copy_bytes_per_token": 0,
        "page_lookups_per_token": 0,
        "oom_status": False,
        "slow_status": False,
        "gpu_name": "cpu",
        "cuda_version": "none",
        "pytorch_version": "test",
        "driver_version": "none",
        "commit_hash": "abc123",
    }
    row.update(overrides)
    return row


def test_required_raw_result_fields_are_nonempty():
    assert len(REQUIRED_RAW_RESULT_FIELDS) > 0
    assert "experiment_id" in REQUIRED_RAW_RESULT_FIELDS
    assert "cache_type" in REQUIRED_RAW_RESULT_FIELDS
    assert "tpot_ms" in REQUIRED_RAW_RESULT_FIELDS
    assert "fragmentation_ratio" in REQUIRED_RAW_RESULT_FIELDS
    assert "commit_hash" in REQUIRED_RAW_RESULT_FIELDS


def test_logger_writes_header_and_row(tmp_path):
    output_path = tmp_path / "raw" / "results.csv"
    logger = RawCSVLogger(output_path)

    row = make_complete_row()
    logger.write_rows([row])

    assert output_path.exists()

    with output_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert reader.fieldnames == REQUIRED_RAW_RESULT_FIELDS
    assert len(rows) == 1
    assert rows[0]["experiment_id"] == "h1_uniform"
    assert rows[0]["cache_type"] == "contiguous"
    assert rows[0]["commit_hash"] == "abc123"


def test_logger_rejects_missing_required_field(tmp_path):
    output_path = tmp_path / "results.csv"
    logger = RawCSVLogger(output_path)

    row = make_complete_row()
    del row["commit_hash"]

    with pytest.raises(ValueError):
        logger.write_rows([row])


def test_logger_ignores_extra_fields(tmp_path):
    output_path = tmp_path / "results.csv"
    logger = RawCSVLogger(output_path)

    row = make_complete_row(extra_debug_field="should_not_be_written")
    logger.write_rows([row])

    with output_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert "extra_debug_field" not in reader.fieldnames
    assert len(rows) == 1


def test_logger_appends_rows(tmp_path):
    output_path = tmp_path / "results.csv"
    logger = RawCSVLogger(output_path)

    logger.append_row(make_complete_row(run_id=0))
    logger.append_row(make_complete_row(run_id=1))

    with output_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 2
    assert rows[0]["run_id"] == "0"
    assert rows[1]["run_id"] == "1"
