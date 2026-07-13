import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union


REQUIRED_RAW_RESULT_FIELDS = [
    "experiment_id",
    "cache_type",
    "block_size",
    "workload_type",
    "batch_size",
    "seq_len_min",
    "seq_len_mean",
    "seq_len_max",
    "decode_len",
    "run_id",
    "warmup_runs",
    "measured_runs",
    "tpot_ms",
    "throughput_tok_s",
    "p50_latency_ms",
    "p95_latency_ms",
    "peak_gpu_memory_mb",
    "live_kv_memory_mb",
    "allocated_pages",
    "free_pages",
    "used_tokens",
    "wasted_tokens",
    "fragmentation_ratio",
    "copy_bytes_per_token",
    "page_lookups_per_token",
    "oom_status",
    "slow_status",
    "gpu_name",
    "cuda_version",
    "pytorch_version",
    "driver_version",
    "commit_hash",
]


class RawCSVLogger:
    """
    Strict CSV logger for raw benchmark results.

    The logger enforces a fixed schema so benchmark rows cannot silently
    omit important fields such as commit hash, memory metrics, OOM status,
    or fragmentation ratio.
    """

    def __init__(
        self,
        output_path: Union[str, Path],
        fieldnames: Optional[List[str]] = None,
    ):
        self.output_path = Path(output_path)
        self.fieldnames = fieldnames or REQUIRED_RAW_RESULT_FIELDS

        if len(self.fieldnames) == 0:
            raise ValueError("fieldnames cannot be empty")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def validate_row(self, row: Dict) -> None:
        missing_fields = [
            field for field in self.fieldnames
            if field not in row
        ]

        if missing_fields:
            raise ValueError(f"row is missing required fields: {missing_fields}")

    def write_rows(self, rows: Iterable[Dict]) -> None:
        rows = list(rows)

        for row in rows:
            self.validate_row(row)

        with self.output_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self.fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)

    def append_row(self, row: Dict) -> None:
        self.validate_row(row)

        file_exists = self.output_path.exists()

        with self.output_path.open("a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self.fieldnames,
                extrasaction="ignore",
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)
