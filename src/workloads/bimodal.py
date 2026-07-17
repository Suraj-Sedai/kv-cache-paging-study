import random
from typing import List, Optional, Tuple

from src.workloads.length_batch import LengthBatch


def _validate_positive_range(name: str, value_range: Tuple[int, int]) -> None:
    low, high = value_range

    if low <= 0:
        raise ValueError(f"{name} lower bound must be positive")

    if high < low:
        raise ValueError(f"{name} upper bound must be >= lower bound")


def _validate_nonnegative_range(name: str, value_range: Tuple[int, int]) -> None:
    low, high = value_range

    if low < 0:
        raise ValueError(f"{name} lower bound must be non-negative")

    if high < low:
        raise ValueError(f"{name} upper bound must be >= lower bound")


def _sample_int_range(
    rng: random.Random,
    value_range: Tuple[int, int],
) -> int:
    low, high = value_range
    return rng.randint(low, high)


def make_bimodal_workload(
    batch_size: int,
    short_prompt_range: Tuple[int, int] = (128, 512),
    long_prompt_range: Tuple[int, int] = (2048, 4096),
    decode_range: Tuple[int, int] = (16, 128),
    short_fraction: float = 0.75,
    seed: Optional[int] = 0,
) -> LengthBatch:
    """
    Create a bimodal workload with many short requests and some long requests.

    This workload is meant to create length variance and fragmentation pressure.
    It is not a production trace. It is a controlled synthetic workload.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if short_fraction <= 0.0 or short_fraction >= 1.0:
        raise ValueError("short_fraction must be between 0 and 1")

    _validate_positive_range("short_prompt_range", short_prompt_range)
    _validate_positive_range("long_prompt_range", long_prompt_range)
    _validate_nonnegative_range("decode_range", decode_range)

    rng = random.Random(seed)

    num_short = int(round(batch_size * short_fraction))
    num_short = max(1, min(batch_size - 1, num_short))
    num_long = batch_size - num_short

    prompt_lengths: List[int] = []
    decode_lengths: List[int] = []

    for _ in range(num_short):
        prompt_lengths.append(_sample_int_range(rng, short_prompt_range))
        decode_lengths.append(_sample_int_range(rng, decode_range))

    for _ in range(num_long):
        prompt_lengths.append(_sample_int_range(rng, long_prompt_range))
        decode_lengths.append(_sample_int_range(rng, decode_range))

    paired = list(zip(prompt_lengths, decode_lengths))
    rng.shuffle(paired)

    prompt_lengths = [prompt for prompt, _ in paired]
    decode_lengths = [decode for _, decode in paired]

    return LengthBatch(
        prompt_lengths=prompt_lengths,
        decode_lengths=decode_lengths,
    )
