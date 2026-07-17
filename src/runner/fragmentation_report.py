import csv
from pathlib import Path
from typing import Dict, Iterable, List, Union

from src.metrics.fragmentation import expected_fragmentation_for_lengths


FRAGMENTATION_REPORT_FIELDS = [
    "workload_type",
    "batch_size",
    "seq_len_min",
    "seq_len_mean",
    "seq_len_max",
    "block_size",
    "total_used_tokens",
    "total_allocated_tokens",
    "total_wasted_tokens",
    "fragmentation_ratio",
    "mean_pages_per_sequence",
    "max_pages_per_sequence",
]


def build_fragmentation_report_rows(
    workload_type: str,
    workload,
    block_sizes: Iterable[int],
) -> List[Dict]:
    """
    Build analytical fragmentation rows for one workload across block sizes.

    This function does not run cache code and does not measure latency.
    It predicts fragmentation from workload lengths and block size.
    """
    if not workload_type:
        raise ValueError("workload_type cannot be empty")

    block_sizes = list(block_sizes)

    if len(block_sizes) == 0:
        raise ValueError("block_sizes cannot be empty")

    rows = []

    for block_size in block_sizes:
        result = expected_fragmentation_for_lengths(
            lengths=workload.total_lengths,
            block_size=block_size,
        )

        pages = result["pages_per_sequence"]

        row = {
            "workload_type": workload_type,
            "batch_size": workload.batch_size,
            "seq_len_min": workload.seq_len_min,
            "seq_len_mean": workload.seq_len_mean,
            "seq_len_max": workload.seq_len_max,
            "block_size": block_size,
            "total_used_tokens": result["total_used_tokens"],
            "total_allocated_tokens": result["total_allocated_tokens"],
            "total_wasted_tokens": result["total_wasted_tokens"],
            "fragmentation_ratio": result["fragmentation_ratio"],
            "mean_pages_per_sequence": sum(pages) / len(pages),
            "max_pages_per_sequence": max(pages),
        }

        rows.append(row)

    return rows


def write_fragmentation_report_csv(
    rows: Iterable[Dict],
    output_path: Union[str, Path],
) -> None:
    """
    Write analytical fragmentation report rows to CSV.
    """
    rows = list(rows)

    if len(rows) == 0:
        raise ValueError("rows cannot be empty")

    for row in rows:
        missing = [
            field for field in FRAGMENTATION_REPORT_FIELDS
            if field not in row
        ]
        if missing:
            raise ValueError(f"row missing required fields: {missing}")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=FRAGMENTATION_REPORT_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
