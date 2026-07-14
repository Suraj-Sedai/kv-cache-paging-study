import pytest

from src.metrics.fragmentation import (
    allocated_tokens_for_sequence,
    expected_fragmentation_for_lengths,
    fragmentation_ratio_for_sequence,
    pages_for_sequence,
    sweep_block_sizes_for_lengths,
    wasted_tokens_for_sequence,
)


@pytest.mark.parametrize(
    "seq_len,block_size,expected_pages",
    [
        (0, 4, 0),
        (1, 4, 1),
        (4, 4, 1),
        (5, 4, 2),
        (8, 4, 2),
        (9, 4, 3),
    ],
)
def test_pages_for_sequence(seq_len, block_size, expected_pages):
    assert pages_for_sequence(seq_len, block_size) == expected_pages


def test_pages_for_sequence_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        pages_for_sequence(-1, 4)

    with pytest.raises(ValueError):
        pages_for_sequence(4, 0)


@pytest.mark.parametrize(
    "seq_len,block_size,expected_allocated,expected_wasted,expected_ratio",
    [
        (0, 4, 0, 0, 0.0),
        (1, 4, 4, 3, 0.75),
        (4, 4, 4, 0, 0.0),
        (5, 4, 8, 3, 0.375),
        (8, 4, 8, 0, 0.0),
        (9, 4, 12, 3, 0.25),
    ],
)
def test_sequence_fragmentation_formulas(
    seq_len,
    block_size,
    expected_allocated,
    expected_wasted,
    expected_ratio,
):
    assert allocated_tokens_for_sequence(seq_len, block_size) == expected_allocated
    assert wasted_tokens_for_sequence(seq_len, block_size) == expected_wasted
    assert fragmentation_ratio_for_sequence(seq_len, block_size) == pytest.approx(
        expected_ratio
    )


def test_expected_fragmentation_for_lengths():
    result = expected_fragmentation_for_lengths(
        lengths=[5, 8, 9],
        block_size=4,
    )

    assert result["block_size"] == 4
    assert result["num_sequences"] == 3

    assert result["total_used_tokens"] == 22

    # 5 -> 8 allocated, 8 -> 8 allocated, 9 -> 12 allocated
    assert result["total_allocated_tokens"] == 28
    assert result["total_wasted_tokens"] == 6
    assert result["fragmentation_ratio"] == pytest.approx(6 / 28)

    assert result["pages_per_sequence"] == [2, 2, 3]
    assert result["allocated_tokens_per_sequence"] == [8, 8, 12]
    assert result["wasted_tokens_per_sequence"] == [3, 0, 3]


def test_expected_fragmentation_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        expected_fragmentation_for_lengths([], block_size=4)

    with pytest.raises(ValueError):
        expected_fragmentation_for_lengths([1, -1], block_size=4)

    with pytest.raises(ValueError):
        expected_fragmentation_for_lengths([1, 2], block_size=0)


def test_sweep_block_sizes_for_lengths():
    results = sweep_block_sizes_for_lengths(
        lengths=[5, 8, 9],
        block_sizes=[2, 4, 8],
    )

    assert len(results) == 3
    assert [result["block_size"] for result in results] == [2, 4, 8]


def test_sweep_block_sizes_rejects_empty_block_sizes():
    with pytest.raises(ValueError):
        sweep_block_sizes_for_lengths(lengths=[5, 8, 9], block_sizes=[])
