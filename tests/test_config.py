import pytest
import yaml

from src.runner.config import (
    UniformMicrobenchmarkConfig,
    load_uniform_microbenchmark_config,
    save_config_snapshot,
)


def make_config_dict(**overrides):
    data = {
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
        "measured_runs": 3,
        "device": "cpu",
        "dtype": "float32",
        "output_path": "results/raw/h1_uniform_micro.csv",
    }
    data.update(overrides)
    return data


def test_uniform_microbenchmark_config_validates_fields():
    config = UniformMicrobenchmarkConfig(**make_config_dict())

    assert config.experiment_id == "h1_uniform_micro"
    assert config.workload_type == "uniform"
    assert config.batch_size == 2
    assert config.prompt_len == 6
    assert config.decode_len == 2
    assert config.block_size == 4
    assert config.dtype == "float32"


def test_uniform_microbenchmark_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        UniformMicrobenchmarkConfig(**make_config_dict(batch_size=0))

    with pytest.raises(ValueError):
        UniformMicrobenchmarkConfig(**make_config_dict(prompt_len=0))

    with pytest.raises(ValueError):
        UniformMicrobenchmarkConfig(**make_config_dict(decode_len=0))

    with pytest.raises(ValueError):
        UniformMicrobenchmarkConfig(**make_config_dict(block_size=0))

    with pytest.raises(ValueError):
        UniformMicrobenchmarkConfig(**make_config_dict(measured_runs=0))

    with pytest.raises(ValueError):
        UniformMicrobenchmarkConfig(**make_config_dict(dtype="int8"))


def test_load_uniform_microbenchmark_config(tmp_path):
    config_path = tmp_path / "h1_uniform_micro.yaml"

    with config_path.open("w") as f:
        yaml.safe_dump(make_config_dict(), f)

    config = load_uniform_microbenchmark_config(config_path)

    assert config.experiment_id == "h1_uniform_micro"
    assert config.batch_size == 2
    assert config.output_path == "results/raw/h1_uniform_micro.csv"


def test_load_uniform_microbenchmark_config_rejects_missing_fields(tmp_path):
    config_path = tmp_path / "bad.yaml"
    data = make_config_dict()
    del data["batch_size"]

    with config_path.open("w") as f:
        yaml.safe_dump(data, f)

    with pytest.raises(ValueError):
        load_uniform_microbenchmark_config(config_path)


def test_load_uniform_microbenchmark_config_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_uniform_microbenchmark_config(tmp_path / "missing.yaml")


def test_save_config_snapshot(tmp_path):
    config = UniformMicrobenchmarkConfig(**make_config_dict())
    snapshot_path = tmp_path / "results" / "raw" / "h1_uniform_micro.config.yaml"

    save_config_snapshot(config, snapshot_path)

    assert snapshot_path.exists()

    with snapshot_path.open("r") as f:
        data = yaml.safe_load(f)

    assert data["experiment_id"] == "h1_uniform_micro"
    assert data["batch_size"] == 2
    assert data["output_path"] == "results/raw/h1_uniform_micro.csv"
