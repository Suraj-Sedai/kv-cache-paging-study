import pytest
import torch

from src.cache.paged_materialized import PagedMaterializedKVCache
from src.metrics.fragmentation import expected_fragmentation_for_lengths
from src.workloads.bimodal import make_bimodal_workload
from src.workloads.heavy_tail import make_heavy_tail_workload
from src.workloads.uniform import make_uniform_workload


def make_key_value(value: float):
    key = torch.full((1, 2), float(value), dtype=torch.float32)
    value_tensor = torch.full((1, 2), float(value + 1000), dtype=torch.float32)
    return key, value_tensor


def fill_paged_cache_for_lengths(lengths, block_size):
    cache = PagedMaterializedKVCache(
        num_layers=1,
        max_batch_size=len(lengths),
        max_seq_len=max(lengths),
        num_kv_heads=1,
        head_dim=2,
        block_size=block_size,
        dtype=torch.float32,
        device="cpu",
    )

    for seq_id, seq_len in enumerate(lengths):
        for token_pos in range(seq_len):
            key, value = make_key_value(seq_id * 1000 + token_pos)
            cache.write(
                layer_id=0,
                seq_id=seq_id,
                token_pos=token_pos,
                key=key,
                value=value,
            )

    return cache


@pytest.mark.parametrize(
    "lengths,block_size",
    [
        ([5, 8, 9], 4),
        ([1, 4, 5, 8], 4),
        ([7, 13, 21], 8),
        ([16, 16, 16], 4),
        ([17, 31, 64], 16),
    ],
)
def test_paged_cache_fragmentation_matches_analytical_model(lengths, block_size):
    cache = fill_paged_cache_for_lengths(lengths, block_size)
    actual = cache.fragmentation()
    expected = expected_fragmentation_for_lengths(lengths, block_size)

    assert actual["used_tokens"] == expected["total_used_tokens"]
    assert actual["allocated_tokens"] == expected["total_allocated_tokens"]
    assert actual["wasted_tokens"] == expected["total_wasted_tokens"]
    assert actual["fragmentation_ratio"] == pytest.approx(
        expected["fragmentation_ratio"]
    )


def test_uniform_workload_fragmentation_matches_analytical_model():
    workload = make_uniform_workload(
        batch_size=4,
        prompt_len=8,
        decode_len=0,
    )
    lengths = workload.total_lengths
    block_size = 4

    cache = fill_paged_cache_for_lengths(lengths, block_size)
    actual = cache.fragmentation()
    expected = expected_fragmentation_for_lengths(lengths, block_size)

    assert actual["used_tokens"] == expected["total_used_tokens"]
    assert actual["allocated_tokens"] == expected["total_allocated_tokens"]
    assert actual["wasted_tokens"] == expected["total_wasted_tokens"]
    assert actual["fragmentation_ratio"] == pytest.approx(
        expected["fragmentation_ratio"]
    )


def test_bimodal_workload_fragmentation_matches_analytical_model():
    workload = make_bimodal_workload(
        batch_size=8,
        short_prompt_range=(5, 9),
        long_prompt_range=(17, 25),
        decode_range=(0, 0),
        short_fraction=0.75,
        seed=123,
    )
    lengths = workload.total_lengths
    block_size = 8

    cache = fill_paged_cache_for_lengths(lengths, block_size)
    actual = cache.fragmentation()
    expected = expected_fragmentation_for_lengths(lengths, block_size)

    assert actual["used_tokens"] == expected["total_used_tokens"]
    assert actual["allocated_tokens"] == expected["total_allocated_tokens"]
    assert actual["wasted_tokens"] == expected["total_wasted_tokens"]
    assert actual["fragmentation_ratio"] == pytest.approx(
        expected["fragmentation_ratio"]
    )


def test_heavy_tail_workload_fragmentation_matches_analytical_model():
    workload = make_heavy_tail_workload(
        batch_size=16,
        min_prompt_len=4,
        max_prompt_len=64,
        decode_range=(0, 0),
        alpha=1.4,
        seed=7,
    )
    lengths = workload.total_lengths
    block_size = 8

    cache = fill_paged_cache_for_lengths(lengths, block_size)
    actual = cache.fragmentation()
    expected = expected_fragmentation_for_lengths(lengths, block_size)

    assert actual["used_tokens"] == expected["total_used_tokens"]
    assert actual["allocated_tokens"] == expected["total_allocated_tokens"]
    assert actual["wasted_tokens"] == expected["total_wasted_tokens"]
    assert actual["fragmentation_ratio"] == pytest.approx(
        expected["fragmentation_ratio"]
    )