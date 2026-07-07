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