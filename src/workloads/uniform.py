from src.workloads.length_batch import LengthBatch


def make_uniform_workload(
    batch_size: int,
    prompt_len: int,
    decode_len: int,
) -> LengthBatch:
    """
    Create a uniform workload where all requests have identical lengths.

    This workload is a control experiment. It isolates paging overhead because
    all sequences advance together and have little fragmentation pressure.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if prompt_len <= 0:
        raise ValueError("prompt_len must be positive")

    if decode_len < 0:
        raise ValueError("decode_len must be non-negative")

    return LengthBatch(
        prompt_lengths=[prompt_len] * batch_size,
        decode_lengths=[decode_len] * batch_size,
    )
