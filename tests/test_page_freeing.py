import pytest

from tests.conftest import make_kv, make_paged_cache


def test_paged_free_releases_pages_and_deactivates_sequence():
    paged = make_paged_cache(block_size=4)

    for token_pos in range(6):
        key, value = make_kv(token_pos, token_pos)
        paged.write(layer_id=0, seq_id=0, token_pos=token_pos, key=key, value=value)

    before_free = paged.stats()
    assert before_free["allocated_pages"] == 2
    assert before_free["free_pages"] == 4
    assert before_free["active_sequences"] == 1

    paged.free(seq_id=0)

    after_free = paged.stats()
    assert after_free["allocated_pages"] == 0
    assert after_free["free_pages"] == 6
    assert after_free["active_sequences"] == 0

    with pytest.raises(KeyError):
        paged.read(layer_id=0, seq_id=0, upto_pos=1)


def test_paged_reuses_freed_pages():
    paged = make_paged_cache(block_size=4)

    for token_pos in range(6):
        key, value = make_kv(token_pos, token_pos)
        paged.write(layer_id=0, seq_id=0, token_pos=token_pos, key=key, value=value)

    first_table = list(paged.block_tables[0])
    assert len(first_table) == 2

    paged.free(seq_id=0)

    for token_pos in range(6):
        key, value = make_kv(token_pos + 10, token_pos + 20)
        paged.write(layer_id=0, seq_id=1, token_pos=token_pos, key=key, value=value)

    second_table = list(paged.block_tables[1])
    assert len(second_table) == 2

    assert set(second_table).issubset(set(range(paged.max_pages)))

    stats = paged.stats()
    assert stats["allocated_pages"] == 2
    assert stats["free_pages"] == 4
    assert stats["active_sequences"] == 1


def test_freeing_one_sequence_does_not_free_other_sequence():
    paged = make_paged_cache(block_size=4)

    for token_pos in range(6):
        key0, value0 = make_kv(token_pos, token_pos)
        key1, value1 = make_kv(token_pos + 100, token_pos + 200)

        paged.write(layer_id=0, seq_id=0, token_pos=token_pos, key=key0, value=value0)
        paged.write(layer_id=0, seq_id=1, token_pos=token_pos, key=key1, value=value1)

    before_free = paged.stats()
    assert before_free["allocated_pages"] == 4
    assert before_free["active_sequences"] == 2

    paged.free(seq_id=0)

    after_free = paged.stats()
    assert after_free["allocated_pages"] == 2
    assert after_free["active_sequences"] == 1

    read_key, read_value = paged.read(layer_id=0, seq_id=1, upto_pos=6)
    assert read_key.shape == (2, 6, 4)
    assert read_value.shape == (2, 6, 4)

    with pytest.raises(KeyError):
        paged.read(layer_id=0, seq_id=0, upto_pos=1)