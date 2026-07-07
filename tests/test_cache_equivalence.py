import pytest
import torch

from src.cache.contiguous import ContiguousKVCache
from src.cache.paged_materialized import PagedMaterializedKVCache


def make_contiguous_cache():
    return ContiguousKVCache(
        num_layers=2,
        max_batch_size=3,
        max_seq_len=8,
        num_kv_heads=2,
        head_dim=4,
        dtype=torch.float32,
        device="cpu",
    )


def make_paged_cache(block_size=4):
    return PagedMaterializedKVCache(
        num_layers=2,
        max_batch_size=3,
        max_seq_len=8,
        num_kv_heads=2,
        head_dim=4,
        block_size=block_size,
        dtype=torch.float32,
        device="cpu",
    )


def make_kv(key_value, value_value):
    key = torch.full((2, 4), float(key_value), dtype=torch.float32)
    value = torch.full((2, 4), float(value_value), dtype=torch.float32)
    return key, value


def assert_cache_reads_equal(contiguous, paged, layer_id, seq_id, upto_pos):
    contig_key, contig_value = contiguous.read(
        layer_id=layer_id,
        seq_id=seq_id,
        upto_pos=upto_pos,
    )
    paged_key, paged_value = paged.read(
        layer_id=layer_id,
        seq_id=seq_id,
        upto_pos=upto_pos,
    )

    assert contig_key.shape == paged_key.shape
    assert contig_value.shape == paged_value.shape

    torch.testing.assert_close(contig_key, paged_key)
    torch.testing.assert_close(contig_value, paged_value)


def write_same_token(contiguous, paged, layer_id, seq_id, token_pos, key_value, value_value):
    key, value = make_kv(key_value, value_value)

    contiguous.write(
        layer_id=layer_id,
        seq_id=seq_id,
        token_pos=token_pos,
        key=key,
        value=value,
    )
    paged.write(
        layer_id=layer_id,
        seq_id=seq_id,
        token_pos=token_pos,
        key=key,
        value=value,
    )


def test_single_token_equivalence():
    contiguous = make_contiguous_cache()
    paged = make_paged_cache(block_size=4)

    write_same_token(
        contiguous,
        paged,
        layer_id=0,
        seq_id=0,
        token_pos=0,
        key_value=1.0,
        value_value=2.0,
    )

    assert_cache_reads_equal(contiguous, paged, layer_id=0, seq_id=0, upto_pos=1)


def test_multi_token_equivalence_across_block_boundary():
    contiguous = make_contiguous_cache()
    paged = make_paged_cache(block_size=4)

    for token_pos in range(8):
        write_same_token(
            contiguous,
            paged,
            layer_id=0,
            seq_id=0,
            token_pos=token_pos,
            key_value=token_pos + 10,
            value_value=token_pos + 20,
        )

    assert_cache_reads_equal(contiguous, paged, layer_id=0, seq_id=0, upto_pos=8)

    paged_stats = paged.stats()
    assert paged_stats["allocated_pages"] == 2
    assert paged_stats["page_lookups"] >= 2


def test_partial_read_excludes_unrequested_tokens():
    contiguous = make_contiguous_cache()
    paged = make_paged_cache(block_size=4)

    for token_pos in range(8):
        write_same_token(
            contiguous,
            paged,
            layer_id=0,
            seq_id=0,
            token_pos=token_pos,
            key_value=token_pos,
            value_value=token_pos + 100,
        )

    contig_key, contig_value = contiguous.read(layer_id=0, seq_id=0, upto_pos=5)
    paged_key, paged_value = paged.read(layer_id=0, seq_id=0, upto_pos=5)

    assert contig_key.shape == (2, 5, 4)
    assert contig_value.shape == (2, 5, 4)
    assert paged_key.shape == (2, 5, 4)
    assert paged_value.shape == (2, 5, 4)

    torch.testing.assert_close(contig_key, paged_key)
    torch.testing.assert_close(contig_value, paged_value)


def test_read_beyond_written_tokens_rejected_by_both_caches():
    contiguous = make_contiguous_cache()
    paged = make_paged_cache(block_size=4)

    write_same_token(
        contiguous,
        paged,
        layer_id=0,
        seq_id=0,
        token_pos=0,
        key_value=1.0,
        value_value=2.0,
    )

    with pytest.raises(ValueError):
        contiguous.read(layer_id=0, seq_id=0, upto_pos=2)

    with pytest.raises(ValueError):
        paged.read(layer_id=0, seq_id=0, upto_pos=2)


def test_sequence_isolation_equivalence():
    contiguous = make_contiguous_cache()
    paged = make_paged_cache(block_size=4)

    write_same_token(
        contiguous,
        paged,
        layer_id=0,
        seq_id=0,
        token_pos=0,
        key_value=1.0,
        value_value=2.0,
    )
    write_same_token(
        contiguous,
        paged,
        layer_id=0,
        seq_id=1,
        token_pos=0,
        key_value=10.0,
        value_value=20.0,
    )

    assert_cache_reads_equal(contiguous, paged, layer_id=0, seq_id=0, upto_pos=1)
    assert_cache_reads_equal(contiguous, paged, layer_id=0, seq_id=1, upto_pos=1)

    paged_key_0, paged_value_0 = paged.read(layer_id=0, seq_id=0, upto_pos=1)
    paged_key_1, paged_value_1 = paged.read(layer_id=0, seq_id=1, upto_pos=1)

    assert not torch.equal(paged_key_0, paged_key_1)
    assert not torch.equal(paged_value_0, paged_value_1)


def test_layer_isolation_equivalence():
    contiguous = make_contiguous_cache()
    paged = make_paged_cache(block_size=4)

    write_same_token(
        contiguous,
        paged,
        layer_id=0,
        seq_id=0,
        token_pos=0,
        key_value=1.0,
        value_value=2.0,
    )
    write_same_token(
        contiguous,
        paged,
        layer_id=1,
        seq_id=0,
        token_pos=0,
        key_value=30.0,
        value_value=40.0,
    )

    assert_cache_reads_equal(contiguous, paged, layer_id=0, seq_id=0, upto_pos=1)
    assert_cache_reads_equal(contiguous, paged, layer_id=1, seq_id=0, upto_pos=1)

    paged_key_0, paged_value_0 = paged.read(layer_id=0, seq_id=0, upto_pos=1)
    paged_key_1, paged_value_1 = paged.read(layer_id=1, seq_id=0, upto_pos=1)

    assert not torch.equal(paged_key_0, paged_key_1)
    assert not torch.equal(paged_value_0, paged_value_1)


def test_paged_fragmentation_matches_formula():
    paged = make_paged_cache(block_size=4)

    for token_pos in range(6):
        key, value = make_kv(token_pos, token_pos)
        paged.write(
            layer_id=0,
            seq_id=0,
            token_pos=token_pos,
            key=key,
            value=value,
        )

    frag = paged.fragmentation()

    assert frag["used_tokens"] == 6
    assert frag["allocated_tokens"] == 8
    assert frag["wasted_tokens"] == 2
    assert frag["fragmentation_ratio"] == pytest.approx(0.25)


def test_paged_free_releases_pages_and_deactivates_sequence():
    paged = make_paged_cache(block_size=4)

    for token_pos in range(6):
        key, value = make_kv(token_pos, token_pos)
        paged.write(
            layer_id=0,
            seq_id=0,
            token_pos=token_pos,
            key=key,
            value=value,
        )

    before_free = paged.stats()
    assert before_free["allocated_pages"] == 2
    assert before_free["free_pages"] == 4

    paged.free(seq_id=0)

    after_free = paged.stats()
    assert after_free["allocated_pages"] == 0
    assert after_free["free_pages"] == 6
    assert after_free["active_sequences"] == 0

    with pytest.raises(KeyError):
        paged.read(layer_id=0, seq_id=0, upto_pos=1)


def test_different_block_sizes_preserve_equivalence():
    for block_size in [1, 2, 4, 8]:
        contiguous = make_contiguous_cache()
        paged = make_paged_cache(block_size=block_size)

        for token_pos in range(7):
            write_same_token(
                contiguous,
                paged,
                layer_id=0,
                seq_id=0,
                token_pos=token_pos,
                key_value=token_pos + 1,
                value_value=token_pos + 50,
            )

        assert_cache_reads_equal(contiguous, paged, layer_id=0, seq_id=0, upto_pos=7)