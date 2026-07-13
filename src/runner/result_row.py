from typing import Dict, Optional

from src.metrics.hardware import raw_csv_hardware_fields
from src.runner.logger import REQUIRED_RAW_RESULT_FIELDS


def _bytes_to_mb(num_bytes: int) -> float:
    if num_bytes < 0:
        raise ValueError("num_bytes must be non-negative")

    return num_bytes / (1024.0 * 1024.0)


def build_raw_result_row(
    experiment_id: str,
    cache,
    workload,
    cache_type: str,
    block_size: int,
    workload_type: str,
    decode_len: int,
    run_id: int,
    warmup_runs: int,
    measured_runs: int,
    tpot_ms: float,
    throughput_tok_s: float,
    p50_latency_ms: float,
    p95_latency_ms: float,
    peak_gpu_memory_mb: float,
    oom_status: bool,
    slow_status: bool,
    hardware_fields: Optional[Dict[str, str]] = None,
) -> Dict:
    """
    Build one raw benchmark result row using the fixed CSV schema.

    This function does not run a benchmark. It formats evidence from:
    - workload metadata,
    - cache memory stats,
    - fragmentation stats,
    - timing measurements,
    - hardware/software metadata,
    - commit hash.
    """
    if not experiment_id:
        raise ValueError("experiment_id cannot be empty")

    if not cache_type:
        raise ValueError("cache_type cannot be empty")

    if not workload_type:
        raise ValueError("workload_type cannot be empty")

    if block_size < 0:
        raise ValueError("block_size must be non-negative")

    if decode_len < 0:
        raise ValueError("decode_len must be non-negative")

    if run_id < 0:
        raise ValueError("run_id must be non-negative")

    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative")

    if measured_runs <= 0:
        raise ValueError("measured_runs must be positive")

    if tpot_ms < 0:
        raise ValueError("tpot_ms must be non-negative")

    if throughput_tok_s < 0:
        raise ValueError("throughput_tok_s must be non-negative")

    if p50_latency_ms < 0:
        raise ValueError("p50_latency_ms must be non-negative")

    if p95_latency_ms < 0:
        raise ValueError("p95_latency_ms must be non-negative")

    if peak_gpu_memory_mb < 0:
        raise ValueError("peak_gpu_memory_mb must be non-negative")

    cache_stats = cache.stats()
    fragmentation = cache.fragmentation()
    hardware = hardware_fields or raw_csv_hardware_fields()

    copy_bytes = cache_stats.get("copy_bytes", 0)
    page_lookups = cache_stats.get("page_lookups", 0)

    total_output_tokens = workload.batch_size * decode_len

    if total_output_tokens > 0:
        copy_bytes_per_token = copy_bytes / total_output_tokens
        page_lookups_per_token = page_lookups / total_output_tokens
    else:
        copy_bytes_per_token = 0.0
        page_lookups_per_token = 0.0

    row = {
        "experiment_id": experiment_id,
        "cache_type": cache_type,
        "block_size": block_size,
        "workload_type": workload_type,
        "batch_size": workload.batch_size,
        "seq_len_min": workload.seq_len_min,
        "seq_len_mean": workload.seq_len_mean,
        "seq_len_max": workload.seq_len_max,
        "decode_len": decode_len,
        "run_id": run_id,
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "tpot_ms": tpot_ms,
        "throughput_tok_s": throughput_tok_s,
        "p50_latency_ms": p50_latency_ms,
        "p95_latency_ms": p95_latency_ms,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "live_kv_memory_mb": _bytes_to_mb(cache_stats["live_kv_memory_bytes"]),
        "allocated_pages": cache_stats.get("allocated_pages", 0),
        "free_pages": cache_stats.get("free_pages", 0),
        "used_tokens": fragmentation["used_tokens"],
        "wasted_tokens": fragmentation["wasted_tokens"],
        "fragmentation_ratio": fragmentation["fragmentation_ratio"],
        "copy_bytes_per_token": copy_bytes_per_token,
        "page_lookups_per_token": page_lookups_per_token,
        "oom_status": oom_status,
        "slow_status": slow_status,
        "gpu_name": hardware["gpu_name"],
        "cuda_version": hardware["cuda_version"],
        "pytorch_version": hardware["pytorch_version"],
        "driver_version": hardware["driver_version"],
        "commit_hash": hardware["commit_hash"],
    }

    missing = [field for field in REQUIRED_RAW_RESULT_FIELDS if field not in row]
    if missing:
        raise RuntimeError(f"raw result row missing fields: {missing}")

    return row
