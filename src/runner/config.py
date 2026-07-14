from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Union

import yaml


@dataclass(frozen=True)
class UniformMicrobenchmarkConfig:
    experiment_id: str
    workload_type: str

    batch_size: int
    prompt_len: int
    decode_len: int

    num_layers: int
    num_kv_heads: int
    head_dim: int

    block_size: int

    warmup_runs: int
    measured_runs: int

    device: str
    dtype: str

    output_path: str

    def __post_init__(self):
        if not self.experiment_id:
            raise ValueError("experiment_id cannot be empty")

        if self.workload_type != "uniform":
            raise ValueError("workload_type must be 'uniform' for this config")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.prompt_len <= 0:
            raise ValueError("prompt_len must be positive")

        if self.decode_len <= 0:
            raise ValueError("decode_len must be positive")

        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")

        if self.num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")

        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")

        if self.block_size <= 0:
            raise ValueError("block_size must be positive")

        if self.warmup_runs < 0:
            raise ValueError("warmup_runs must be non-negative")

        if self.measured_runs <= 0:
            raise ValueError("measured_runs must be positive")

        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be one of: float32, float16, bfloat16")

        if not self.output_path:
            raise ValueError("output_path cannot be empty")

    def to_dict(self) -> Dict:
        return asdict(self)


REQUIRED_UNIFORM_MICRO_CONFIG_FIELDS = {
    "experiment_id",
    "workload_type",
    "batch_size",
    "prompt_len",
    "decode_len",
    "num_layers",
    "num_kv_heads",
    "head_dim",
    "block_size",
    "warmup_runs",
    "measured_runs",
    "device",
    "dtype",
    "output_path",
}


def load_yaml_config(path: Union[str, Path]) -> Dict:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")

    with config_path.open("r") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"config file is empty: {config_path}")

    if not isinstance(data, dict):
        raise ValueError("config file must contain a mapping/object")

    return data


def load_uniform_microbenchmark_config(
    path: Union[str, Path],
) -> UniformMicrobenchmarkConfig:
    data = load_yaml_config(path)

    missing = REQUIRED_UNIFORM_MICRO_CONFIG_FIELDS - set(data.keys())
    if missing:
        raise ValueError(f"config missing required fields: {sorted(missing)}")

    return UniformMicrobenchmarkConfig(**data)


def save_config_snapshot(
    config: UniformMicrobenchmarkConfig,
    output_path: Union[str, Path],
) -> None:
    """
    Save the exact effective config used for a run.

    This should live next to raw results so future readers can reconstruct
    which settings produced a CSV.
    """
    snapshot_path = Path(output_path)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)

    with snapshot_path.open("w") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=True)
