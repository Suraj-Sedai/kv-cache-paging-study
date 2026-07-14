import csv

import pytest
import yaml

from src.runner.benchmark_runner import (
    _config_snapshot_path_for_output,
    _torch_dtype_from_string,
    run_uniform_cache_microbenchmark,
    run_uniform_cache_microbenchmark_from_config,
)
from src.runner.logger import REQUIRED_RAW_RESULT_FIELDS


def test_torch_dtype_from_string():
    import torch

    assert _torch_dtype_from_string("float32") == torch.float32
    assert _torch_dtype_from_string("float16") == torch.float16
    assert _torch_dtype_from_string("bfloat16") == torch.bfloat16

    with pytest.raises(ValueError):
        _torch_dtype_from_string("int8")


def test_config_snapshot_path_for_output():
    path = _config_snapshot_path_for_output("results/raw/h1_uniform_micro.csv")

    assert str(path) == "results/raw/h1_uniform_micro.config.yaml"


def test_uniform_cache_microbenchmark_writes_valid_raw_csv(tmp_path):
    output_path = tmp_path / "raw" / "h1_uniform_micro.csv"

    rows = run_uniform_cache_microbenchmark(
        output_path=str(output_path),
        batch_size=2,
        prompt_len=6,
        decode_len=2,
        num_layers=2,
        num_kv_heads=2,
        head_dim=4,
        block_size=4,
        warmup_runs=1,
        measured_runs=2,
        device="cpu",
    )

    assert output_path.exists()
    assert len(rows) == 2

    cache_types = {row["cache_type"] for row in rows}
    assert cache_types == {"contiguous", "paged_materialized"}

    for row in rows:
        assert list(row.keys()) == REQUIRED_RAW_RESULT_FIELDS
        assert row["experiment_id"] == "h1_uniform_micro"
        assert row["workload_type"] == "uniform"
        assert row["batch_size"] == 2
        assert row["decode_len"] == 2
        assert row["warmup_runs"] == 1
        assert row["measured_runs"] == 2
        assert row["tpot_ms"] >= 0.0
        assert row["throughput_tok_s"] > 0.0
        assert row["oom_status"] is False
        assert row["slow_status"] is False

    with output_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        csv_rows = list(reader)

    assert reader.fieldnames == REQUIRED_RAW_RESULT_FIELDS
    assert len(csv_rows) == 2


def test_uniform_cache_microbenchmark_rejects_zero_decode_len(tmp_path):
    output_path = tmp_path / "raw" / "bad.csv"

    with pytest.raises(ValueError):
        run_uniform_cache_microbenchmark(
            output_path=str(output_path),
            decode_len=0,
        )


def test_uniform_cache_microbenchmark_rejects_invalid_measured_runs(tmp_path):
    output_path = tmp_path / "raw" / "bad.csv"

    with pytest.raises(ValueError):
        run_uniform_cache_microbenchmark(
            output_path=str(output_path),
            measured_runs=0,
        )


def test_uniform_cache_microbenchmark_from_config_writes_csv_and_config_snapshot(
    tmp_path,
):
    output_path = tmp_path / "raw" / "h1_uniform_micro.csv"
    config_path = tmp_path / "h1_uniform_micro.yaml"

    config = {
        "experiment_id": "h1_uniform_micro",
        "workload_type": "uniform",
        "batch_size": 2,
        "prompt_len": 6,
        "decode_len": 2,
        "num_layers": 2,
        "num_kv_heads": 2,
        "head_dim": 4,
        "block_size": 4,
        "warmup_runs": 1,
        "measured_runs": 2,
        "device": "cpu",
        "dtype": "float32",
        "output_path": str(output_path),
    }

    with config_path.open("w") as f:
        yaml.safe_dump(config, f)

    rows = run_uniform_cache_microbenchmark_from_config(str(config_path))

    snapshot_path = output_path.with_suffix(".config.yaml")

    assert output_path.exists()
    assert snapshot_path.exists()
    assert len(rows) == 2

    with snapshot_path.open("r") as f:
        snapshot = yaml.safe_load(f)

    assert snapshot["experiment_id"] == "h1_uniform_micro"
    assert snapshot["batch_size"] == 2
    assert snapshot["prompt_len"] == 6
    assert snapshot["decode_len"] == 2
    assert snapshot["output_path"] == str(output_path)
