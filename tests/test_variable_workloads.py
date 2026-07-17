import pytest

from src.workloads.bimodal import make_bimodal_workload
from src.workloads.heavy_tail import make_heavy_tail_workload


def test_bimodal_workload_has_expected_batch_size():
    workload = make_bimodal_workload(
        batch_size=8,
        short_prompt_range=(10, 20),
        long_prompt_range=(100, 200),
        decode_range=(1, 4),
        short_fraction=0.75,
        seed=123,
    )

    assert workload.batch_size == 8
    assert len(workload.prompt_lengths) == 8
    assert len(workload.decode_lengths) == 8
    assert workload.seq_len_min > 0
    assert workload.seq_len_max > workload.seq_len_min


def test_bimodal_workload_is_reproducible_with_seed():
    workload_a = make_bimodal_workload(batch_size=10, seed=42)
    workload_b = make_bimodal_workload(batch_size=10, seed=42)

    assert workload_a.prompt_lengths == workload_b.prompt_lengths
    assert workload_a.decode_lengths == workload_b.decode_lengths


def test_bimodal_workload_contains_short_and_long_requests():
    workload = make_bimodal_workload(
        batch_size=10,
        short_prompt_range=(10, 20),
        long_prompt_range=(100, 200),
        decode_range=(1, 1),
        short_fraction=0.8,
        seed=0,
    )

    short_count = sum(10 <= length <= 20 for length in workload.prompt_lengths)
    long_count = sum(100 <= length <= 200 for length in workload.prompt_lengths)

    assert short_count == 8
    assert long_count == 2


def test_bimodal_workload_allows_zero_decode_range():
    workload = make_bimodal_workload(
        batch_size=8,
        short_prompt_range=(10, 20),
        long_prompt_range=(100, 200),
        decode_range=(0, 0),
        short_fraction=0.75,
        seed=123,
    )

    assert workload.batch_size == 8
    assert workload.decode_lengths == [0] * 8


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"short_fraction": 0.0},
        {"short_fraction": 1.0},
        {"short_prompt_range": (0, 10)},
        {"long_prompt_range": (100, 50)},
        {"decode_range": (10, 1)},
    ],
)
def test_bimodal_workload_rejects_invalid_inputs(kwargs):
    call_kwargs = {"batch_size": 8}
    call_kwargs.update(kwargs)

    with pytest.raises(ValueError):
        make_bimodal_workload(**call_kwargs)


def test_heavy_tail_workload_has_expected_batch_size():
    workload = make_heavy_tail_workload(
        batch_size=16,
        min_prompt_len=8,
        max_prompt_len=512,
        decode_range=(1, 4),
        alpha=1.5,
        seed=123,
    )

    assert workload.batch_size == 16
    assert len(workload.prompt_lengths) == 16
    assert len(workload.decode_lengths) == 16
    assert min(workload.prompt_lengths) >= 8
    assert max(workload.prompt_lengths) <= 512


def test_heavy_tail_workload_is_reproducible_with_seed():
    workload_a = make_heavy_tail_workload(batch_size=16, seed=42)
    workload_b = make_heavy_tail_workload(batch_size=16, seed=42)

    assert workload_a.prompt_lengths == workload_b.prompt_lengths
    assert workload_a.decode_lengths == workload_b.decode_lengths


def test_heavy_tail_workload_has_length_variance():
    workload = make_heavy_tail_workload(
        batch_size=64,
        min_prompt_len=8,
        max_prompt_len=4096,
        decode_range=(1, 1),
        alpha=1.3,
        seed=7,
    )

    assert workload.seq_len_max > workload.seq_len_min
    assert workload.seq_len_mean > workload.seq_len_min


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"min_prompt_len": 0},
        {"max_prompt_len": 4, "min_prompt_len": 8},
        {"decode_range": (-1, 4)},
        {"decode_range": (4, 1)},
        {"alpha": 1.0},
    ],
)
def test_heavy_tail_workload_rejects_invalid_inputs(kwargs):
    call_kwargs = {"batch_size": 8}
    call_kwargs.update(kwargs)

    with pytest.raises(ValueError):
        make_heavy_tail_workload(**call_kwargs)
