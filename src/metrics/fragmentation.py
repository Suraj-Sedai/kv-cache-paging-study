import math
from typing import Iterable, List


def pages_for_sequence(seq_len: int, block_size: int) -> int:
    """
    Return the number of pages needed for one sequence.

    Formula:
        pages = ceil(seq_len / block_size)
    """
    if seq_len < 0:
        raise ValueError("seq_len must be non-negative")

    if block_size <= 0:
        raise ValueError("block_size must be positive")

    if seq_len == 0:
        return 0

    return int(math.ceil(seq_len / block_size))


def allocated_tokens_for_sequence(seq_len: int, block_size: int) -> int:
    """
    Return allocated token capacity for one sequence.
    """
    return pages_for_sequence(seq_len, block_size) * block_size


def wasted_tokens_for_sequence(seq_len: int, block_size: int) -> int:
    """
    Return internal fragmentation in tokens for one sequence.
    """
    allocated_tokens = allocated_tokens_for_sequence(seq_len, block_size)
    return allocated_tokens - seq_len


def fragmentation_ratio_for_sequence(seq_len: int, block_size: int) -> float:
    """
    Return internal fragmentation ratio for one sequence.
    """
    allocated_tokens = allocated_tokens_for_sequence(seq_len, block_size)

    if allocated_tokens == 0:
        return 0.0

    wasted_tokens = allocated_tokens - seq_len
    return wasted_tokens / allocated_tokens


def expected_fragmentation_for_lengths(
    lengths: Iterable[int],
    block_size: int,
) -> dict:
    """
    Compute expected fragmentation over a workload.

    For lengths L_i and block size B:

        total_wasted = Σ(ceil(L_i / B) × B - L_i)
        total_allocated = Σ(ceil(L_i / B) × B)
        expected_fragmentation = total_wasted / total_allocated
    """
    lengths = list(lengths)

    if len(lengths) == 0:
        raise ValueError("lengths cannot be empty")

    if any(length < 0 for length in lengths):
        raise ValueError("lengths cannot contain negative values")

    if block_size <= 0:
        raise ValueError("block_size must be positive")

    allocated_per_sequence: List[int] = [
        allocated_tokens_for_sequence(length, block_size)
        for length in lengths
    ]

    wasted_per_sequence: List[int] = [
        wasted_tokens_for_sequence(length, block_size)
        for length in lengths
    ]

    total_used = sum(lengths)
    total_allocated = sum(allocated_per_sequence)
    total_wasted = sum(wasted_per_sequence)

    if total_allocated == 0:
        fragmentation_ratio = 0.0
    else:
        fragmentation_ratio = total_wasted / total_allocated

    return {
        "block_size": block_size,
        "num_sequences": len(lengths),
        "total_used_tokens": total_used,
        "total_allocated_tokens": total_allocated,
        "total_wasted_tokens": total_wasted,
        "fragmentation_ratio": fragmentation_ratio,
        "pages_per_sequence": [
            pages_for_sequence(length, block_size)
            for length in lengths
        ],
        "allocated_tokens_per_sequence": allocated_per_sequence,
        "wasted_tokens_per_sequence": wasted_per_sequence,
    }


def sweep_block_sizes_for_lengths(
    lengths: Iterable[int],
    block_sizes: Iterable[int],
) -> List[dict]:
    """
    Compute expected fragmentation for multiple block sizes.
    """
    lengths = list(lengths)
    block_sizes = list(block_sizes)

    if len(block_sizes) == 0:
        raise ValueError("block_sizes cannot be empty")

    return [
        expected_fragmentation_for_lengths(lengths, block_size)
        for block_size in block_sizes
    ]
