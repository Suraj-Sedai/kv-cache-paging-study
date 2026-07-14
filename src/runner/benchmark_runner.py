from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch

from src.cache.contiguous import ContiguousKVCache
from src.cache.paged_materialized import PagedMaterializedKVCache
from src.metrics.latency import (
    measure_elapsed_ms,
    summarize_latencies_ms,
    throughput_tokens_per_second,
    time_per_output_token_ms,
)
from src.runner.config import (
    UniformMicrobenchmarkConfig,
    load_uniform_microbenchmark_config,
    save_config_snapshot,
)
from src.runner.logger import RawCSVLogger
from src.runner.result_row import build_raw_result_row
from src.workloads.uniform import make_uniform_workload


def _torch_dtype_from_string(dtype_name: str):
    if dtype_name == "float32":
        return torch.float32

    if dtype_name == "float16":
        return torch.float16

    if dtype_name == "bfloat16":
        return torch.bfloat16

    raise ValueError(f"unsupported dtype: {dtype_name}")


def _make_deterministic_kv(
    num_kv_heads: int,
    head_dim: int,
    value: float,
    dtype,
    device: Union[str, torch.device],
) -> Tuple[torch.Tensor, torch.Tensor]:
    key = torch.full(
        (num_kv_heads, head_dim),
        float(value),
        dtype=dtype,
        device=device,
    )
    val = torch.full(
        (num_kv_heads, head_dim),
        float(value + 1000.0),
        dtype=dtype,
        device=device,
    )
    return key, val


def _make_cache(
    cache_type: str,
    num_layers: int,
    batch_size: int,
    max_seq_len: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    dtype,
    device: Union[str, torch.device],
):
    if cache_type == "contiguous":
        return ContiguousKVCache(
            num_layers=num_layers,
            max_batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )

    if cache_type == "paged_materialized":
        return PagedMaterializedKVCache(
            num_layers=num_layers,
            max_batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            dtype=dtype,
            device=device,
        )

    raise ValueError(f"unknown cache_type: {cache_type}")


def _prefill_cache(
    cache,
    workload,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype,
    device: Union[str, torch.device],
) -> None:
    """
    Write prompt tokens into the cache before decode measurement.

    This simulates the cache state after prefill. The timing benchmark below
    measures decode-like cache reads/writes, not the prefill stage.
    """
    for seq_id, prompt_len in enumerate(workload.prompt_lengths):
        for token_pos in range(prompt_len):
            for layer_id in range(num_layers):
                value = layer_id * 100000 + seq_id * 1000 + token_pos
                key, val = _make_deterministic_kv(
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    value=value,
                    dtype=dtype,
                    device=device,
                )
                cache.write(
                    layer_id=layer_id,
                    seq_id=seq_id,
                    token_pos=token_pos,
                    key=key,
                    value=val,
                )


def _run_decode_cache_ops(
    cache,
    workload,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype,
    device: Union[str, torch.device],
) -> None:
    """
    Simulate decode-stage cache operations.

    For each generated token:
      1. read existing KV up to current position,
      2. write the new token's KV.

    This intentionally stresses the read path, where paged materialized cache
    pays block-table lookup and copy/materialization cost.
    """
    for decode_step in range(max(workload.decode_lengths)):
        for seq_id, prompt_len in enumerate(workload.prompt_lengths):
            decode_len = workload.decode_lengths[seq_id]

            if decode_step >= decode_len:
                continue

            token_pos = prompt_len + decode_step

            for layer_id in range(num_layers):
                cache.read(
                    layer_id=layer_id,
                    seq_id=seq_id,
                    upto_pos=token_pos,
                )

                value = layer_id * 100000 + seq_id * 1000 + token_pos
                key, val = _make_deterministic_kv(
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    value=value,
                    dtype=dtype,
                    device=device,
                )

                cache.write(
                    layer_id=layer_id,
                    seq_id=seq_id,
                    token_pos=token_pos,
                    key=key,
                    value=val,
                )


def _reset_peak_memory_if_cuda(device: Union[str, torch.device]) -> None:
    torch_device = torch.device(device)

    if torch_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(torch_device)


def _peak_memory_mb_if_cuda(device: Union[str, torch.device]) -> float:
    torch_device = torch.device(device)

    if torch_device.type != "cuda" or not torch.cuda.is_available():
        return 0.0

    return torch.cuda.max_memory_allocated(torch_device) / (1024.0 * 1024.0)


def _measure_one_cache_type(
    cache_type: str,
    workload,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    warmup_runs: int,
    measured_runs: int,
    dtype,
    device: Union[str, torch.device],
) -> Tuple[object, Dict[str, float]]:
    """
    Run warmup and measured decode-cache-operation trials for one cache type.

    Returns:
      - the final measured cache object, used for stats/fragmentation,
      - latency summary dictionary.
    """
    if measured_runs <= 0:
        raise ValueError("measured_runs must be positive")

    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative")

    max_seq_len = max(workload.total_lengths)

    for _ in range(warmup_runs):
        warmup_cache = _make_cache(
            cache_type=cache_type,
            num_layers=num_layers,
            batch_size=workload.batch_size,
            max_seq_len=max_seq_len,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            dtype=dtype,
            device=device,
        )
        _prefill_cache(
            cache=warmup_cache,
            workload=workload,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )
        _run_decode_cache_ops(
            cache=warmup_cache,
            workload=workload,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )

    latencies_ms: List[float] = []
    final_cache = None

    for _ in range(measured_runs):
        cache = _make_cache(
            cache_type=cache_type,
            num_layers=num_layers,
            batch_size=workload.batch_size,
            max_seq_len=max_seq_len,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            dtype=dtype,
            device=device,
        )
        _prefill_cache(
            cache=cache,
            workload=workload,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )

        _reset_peak_memory_if_cuda(device)

        _, elapsed_ms = measure_elapsed_ms(
            lambda: _run_decode_cache_ops(
                cache=cache,
                workload=workload,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dtype=dtype,
                device=device,
            ),
            device=device,
        )

        latencies_ms.append(elapsed_ms)
        final_cache = cache

    return final_cache, summarize_latencies_ms(latencies_ms)


def run_uniform_cache_microbenchmark(
    output_path: str,
    batch_size: int = 2,
    prompt_len: int = 6,
    decode_len: int = 2,
    num_layers: int = 2,
    num_kv_heads: int = 2,
    head_dim: int = 4,
    block_size: int = 4,
    warmup_runs: int = 1,
    measured_runs: int = 3,
    dtype=torch.float32,
    device: Union[str, torch.device] = "cpu",
) -> List[Dict]:
    """
    Run a minimal uniform cache microbenchmark and write raw CSV rows.

    This is a cache-operation benchmark only. It is useful for validating the
    measurement pipeline before larger experiments.
    """
    if decode_len <= 0:
        raise ValueError("decode_len must be positive for benchmarking")

    workload = make_uniform_workload(
        batch_size=batch_size,
        prompt_len=prompt_len,
        decode_len=decode_len,
    )

    rows = []
    total_output_tokens = workload.batch_size * decode_len

    for cache_type in ["contiguous", "paged_materialized"]:
        effective_block_size = 0 if cache_type == "contiguous" else block_size

        cache, latency_summary = _measure_one_cache_type(
            cache_type=cache_type,
            workload=workload,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
            dtype=dtype,
            device=device,
        )

        mean_elapsed_ms = latency_summary["mean_ms"]

        row = build_raw_result_row(
            experiment_id="h1_uniform_micro",
            cache=cache,
            workload=workload,
            cache_type=cache_type,
            block_size=effective_block_size,
            workload_type="uniform",
            decode_len=decode_len,
            run_id=0,
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
            tpot_ms=time_per_output_token_ms(
                total_elapsed_ms=mean_elapsed_ms,
                num_output_tokens=total_output_tokens,
            ),
            throughput_tok_s=throughput_tokens_per_second(
                num_tokens=total_output_tokens,
                total_elapsed_ms=mean_elapsed_ms,
            ),
            p50_latency_ms=latency_summary["p50_ms"],
            p95_latency_ms=latency_summary["p95_ms"],
            peak_gpu_memory_mb=_peak_memory_mb_if_cuda(device),
            oom_status=False,
            slow_status=False,
        )

        rows.append(row)

    logger = RawCSVLogger(output_path)
    logger.write_rows(rows)

    return rows


def _config_snapshot_path_for_output(output_path: str) -> Path:
    """
    Return the config snapshot path paired with a raw CSV output path.

    Example:
        results/raw/h1_uniform_micro.csv
    becomes:
        results/raw/h1_uniform_micro.config.yaml
    """
    path = Path(output_path)
    return path.with_suffix(".config.yaml")


def run_uniform_cache_microbenchmark_from_config(
    config_path: str,
):
    """
    Load a uniform microbenchmark config, run the benchmark, and save
    a config snapshot next to the raw CSV.

    This is the preferred entry point for reproducible benchmark runs.
    """
    config = load_uniform_microbenchmark_config(config_path)

    rows = run_uniform_cache_microbenchmark(
        output_path=config.output_path,
        batch_size=config.batch_size,
        prompt_len=config.prompt_len,
        decode_len=config.decode_len,
        num_layers=config.num_layers,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        block_size=config.block_size,
        warmup_runs=config.warmup_runs,
        measured_runs=config.measured_runs,
        dtype=_torch_dtype_from_string(config.dtype),
        device=config.device,
    )

    save_config_snapshot(
        config=config,
        output_path=_config_snapshot_path_for_output(config.output_path),
    )

    return rows
