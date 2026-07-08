from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class LengthBatch:
    """
    Represents a batch of request lengths for cache/workload experiments.

    prompt_lengths:
        Initial prompt/context length for each request.

    decode_lengths:
        Number of output tokens to generate for each request.

    total_lengths:
        prompt_lengths[i] + decode_lengths[i]
    """

    prompt_lengths: List[int]
    decode_lengths: List[int]

    def __post_init__(self):
        if len(self.prompt_lengths) == 0:
            raise ValueError("prompt_lengths cannot be empty")

        if len(self.prompt_lengths) != len(self.decode_lengths):
            raise ValueError("prompt_lengths and decode_lengths must have same length")

        if any(length <= 0 for length in self.prompt_lengths):
            raise ValueError("all prompt lengths must be positive")

        if any(length < 0 for length in self.decode_lengths):
            raise ValueError("all decode lengths must be non-negative")

    @property
    def batch_size(self) -> int:
        return len(self.prompt_lengths)

    @property
    def total_lengths(self) -> List[int]:
        return [
            prompt + decode
            for prompt, decode in zip(self.prompt_lengths, self.decode_lengths)
        ]

    @property
    def seq_len_min(self) -> int:
        return min(self.total_lengths)

    @property
    def seq_len_max(self) -> int:
        return max(self.total_lengths)

    @property
    def seq_len_mean(self) -> float:
        return sum(self.total_lengths) / len(self.total_lengths)

    def to_dict(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "prompt_lengths": list(self.prompt_lengths),
            "decode_lengths": list(self.decode_lengths),
            "total_lengths": self.total_lengths,
            "seq_len_min": self.seq_len_min,
            "seq_len_mean": self.seq_len_mean,
            "seq_len_max": self.seq_len_max,
        }
