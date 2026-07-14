import random
from typing import List, Optional, Tuple

from src.workloads.length_batch import LengthBatch


def make_heavy_tail_workload(
    batch_size: int,
    min_prompt_len: int = 64,
    max_prompt_len: int = 8192,
    decode_range: Tuple[int, int] = (16, 128),
    alpha: float = 1.5,
    seed: Optional[int] = 0,
) -> LengthBatch:
    """
    Create a synthetic heavy-tail workload.

    Most requests are short, but a small number become very long. This is closer
    to serving behavior than uniform workloads, but it is still synthetic.

    alpha controls the tail:
      - smaller alpha => heavier tail,
      - larger alpha => fewer extreme long requests.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if min_prompt_len <= 0:
        raise ValueError("min_prompt_len must be positive")

    if max_prompt_len < min_prompt_len:
        raise ValueError("max_prompt_len must be >= min_prompt_len")

    decode_low, decode_high = decode_range

    if decode_low < 0:
        raise ValueError("decode_range lower bound must be non-negative")

    if decode_high < decode_low:
        raise ValueError("decode_range upper bound must be >= lower bound")

    if alpha <= 1.0:
        raise ValueError("alpha must be > 1.0")

    rng = random.Random(seed)

    prompt_lengths: List[int] = []
    decode_lengths: List[int] = []

    for _ in range(batch_size):
        raw = rng.paretovariate(alpha)

        # Convert Pareto sample into a bounded prompt length.
        # Most samples stay near min_prompt_len; rare samples move toward max.
        scaled = int(min_prompt_len * raw)
        prompt_len = min(max(scaled, min_prompt_len), max_prompt_len)

        prompt_lengths.append(prompt_len)
        decode_lengths.append(rng.randint(decode_low, decode_high))

    return LengthBatch(
        prompt_lengths=prompt_lengths,
        decode_lengths=decode_lengths,
    )
