import statistics
import time
from typing import Callable, Iterable, List, Optional, Tuple, TypeVar, Union

import torch


T = TypeVar("T")


def synchronize_if_needed(device: Optional[Union[str, torch.device]] = None) -> None:
    """
    Synchronize CUDA work before/after timing.

    CPU operations are mostly synchronous, so this is a no-op on CPU.
    CUDA operations are asynchronous, so timing without synchronization can
    underestimate latency.
    """
    if not torch.cuda.is_available():
        return

    if device is None:
        torch.cuda.synchronize()
        return

    torch_device = torch.device(device)

    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)


def measure_elapsed_ms(
    fn: Callable[[], T],
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[T, float]:
    """
    Measure wall-clock elapsed time in milliseconds.

    Synchronizes CUDA before and after running fn so asynchronous kernels are
    included in the timing measurement.
    """
    synchronize_if_needed(device)
    start = time.perf_counter()

    result = fn()

    synchronize_if_needed(device)
    end = time.perf_counter()

    elapsed_ms = (end - start) * 1000.0
    return result, elapsed_ms


def summarize_latencies_ms(latencies_ms: Iterable[float]) -> dict:
    """
    Summarize a collection of latency measurements.

    Returns mean, p50, p95, min, max, and count.
    """
    values = list(latencies_ms)

    if len(values) == 0:
        raise ValueError("latencies_ms cannot be empty")

    if any(value < 0 for value in values):
        raise ValueError("latencies_ms cannot contain negative values")

    sorted_values = sorted(values)

    return {
        "count": len(sorted_values),
        "mean_ms": statistics.mean(sorted_values),
        "p50_ms": percentile(sorted_values, 50),
        "p95_ms": percentile(sorted_values, 95),
        "min_ms": sorted_values[0],
        "max_ms": sorted_values[-1],
    }


def percentile(sorted_values: List[float], percentile_value: float) -> float:
    """
    Compute percentile using linear interpolation.

    sorted_values must already be sorted.
    """
    if len(sorted_values) == 0:
        raise ValueError("sorted_values cannot be empty")

    if percentile_value < 0 or percentile_value > 100:
        raise ValueError("percentile_value must be between 0 and 100")

    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (percentile_value / 100.0) * (len(sorted_values) - 1)
    lower_idx = int(rank)
    upper_idx = min(lower_idx + 1, len(sorted_values) - 1)

    lower_value = sorted_values[lower_idx]
    upper_value = sorted_values[upper_idx]

    weight = rank - lower_idx
    return lower_value + weight * (upper_value - lower_value)


def time_per_output_token_ms(total_elapsed_ms: float, num_output_tokens: int) -> float:
    """
    Compute TPOT: time per output token in milliseconds.
    """
    if total_elapsed_ms < 0:
        raise ValueError("total_elapsed_ms must be non-negative")

    if num_output_tokens <= 0:
        raise ValueError("num_output_tokens must be positive")

    return total_elapsed_ms / num_output_tokens


def throughput_tokens_per_second(num_tokens: int, total_elapsed_ms: float) -> float:
    """
    Compute token throughput.
    """
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")

    if total_elapsed_ms <= 0:
        raise ValueError("total_elapsed_ms must be positive")

    return num_tokens / (total_elapsed_ms / 1000.0)
