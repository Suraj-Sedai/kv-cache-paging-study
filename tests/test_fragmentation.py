import pytest

from tests.conftest import make_contiguous_cache, make_kv, make_paged_cache


def test_contiguous_fragmentation_accounting():
    cache = make_contiguous_cache()

    for token_pos in range(6):
        key, value = make_kv(token_pos, token_pos)
        cache.write(layer_id=0, seq_id=0, token_pos=token_pos, key=key, value=value)

    frag = cache.fragmentation()

    assert frag["used_tokens"] == 6
    assert frag["allocated_tokens"] == 8
    assert frag["wasted_tokens"] == 2
    assert frag["fragmentation_ratio"] == pytest.approx(0.25)


def test_paged_fragmentation_matches_formula():
    paged = make_paged_cache(block_size=4)

    for token_pos in range(6):
        key, value = make_kv(token_pos, token_pos)
        paged.write(layer_id=0, seq_id=0, token_pos=token_pos, key=key, value=value)

    frag = paged.fragmentation()

    assert frag["used_tokens"] == 6
    assert frag["allocated_tokens"] == 8
    assert frag["wasted_tokens"] == 2
    assert frag["fragmentation_ratio"] == pytest.approx(0.25)


@pytest.mark.parametrize(
    "seq_len, block_size, expected_allocated, expected_wasted, expected_ratio",
    [
        (1, 4, 4, 3, 0.75),
        (4, 4, 4, 0, 0.0),
        (5, 4, 8, 3, 0.375),
        (8, 4, 8, 0, 0.0),
        (6, 2, 6, 0, 0.0),
        (7, 2, 8, 1, 0.125),
    ],
)
def test_paged_fragmentation_formula_cases(
    seq_len,
    block_size,
    expected_allocated,
    expected_wasted,
    expected_ratio,
):
    paged = make_paged_cache(block_size=block_size)

    for token_pos in range(seq_len):
        key, value = make_kv(token_pos, token_pos)
        paged.write(layer_id=0, seq_id=0, token_pos=token_pos, key=key, value=value)

    frag = paged.fragmentation()

    assert frag["used_tokens"] == seq_len
    assert frag["allocated_tokens"] == expected_allocated
    assert frag["wasted_tokens"] == expected_wasted
    assert frag["fragmentation_ratio"] == pytest.approx(expected_ratio)