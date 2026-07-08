from tests.conftest import make_contiguous_cache, make_kv, make_paged_cache


def test_contiguous_memory_accounting_active_sequence():
    cache = make_contiguous_cache()

    for token_pos in range(6):
        key, value = make_kv(token_pos, token_pos)
        cache.write(layer_id=0, seq_id=0, token_pos=token_pos, key=key, value=value)

    stats = cache.stats()

    # num_layers=2, K+V=2, num_kv_heads=2, head_dim=4, fp32=4 bytes
    per_token_kv_bytes = 2 * 2 * 2 * 4 * 4

    assert stats["live_kv_memory_bytes"] == 6 * per_token_kv_bytes
    assert stats["allocated_kv_memory_bytes"] == 8 * per_token_kv_bytes
    assert stats["reserved_cache_memory_bytes"] == 3 * 8 * per_token_kv_bytes
    assert stats["memory_used_bytes"] == stats["allocated_kv_memory_bytes"]


def test_paged_memory_accounting_active_sequence():
    paged = make_paged_cache(block_size=4)

    for token_pos in range(6):
        key, value = make_kv(token_pos, token_pos)
        paged.write(layer_id=0, seq_id=0, token_pos=token_pos, key=key, value=value)

    stats = paged.stats()

    # num_layers=2, K+V=2, num_kv_heads=2, head_dim=4, fp32=4 bytes
    per_token_kv_bytes = 2 * 2 * 2 * 4 * 4

    assert stats["live_kv_memory_bytes"] == 6 * per_token_kv_bytes
    assert stats["allocated_kv_memory_bytes"] == 8 * per_token_kv_bytes
    assert stats["reserved_cache_memory_bytes"] == 6 * 4 * per_token_kv_bytes
    assert stats["memory_used_bytes"] == stats["allocated_kv_memory_bytes"]


def test_paged_memory_accounting_after_free():
    paged = make_paged_cache(block_size=4)

    for token_pos in range(6):
        key, value = make_kv(token_pos, token_pos)
        paged.write(layer_id=0, seq_id=0, token_pos=token_pos, key=key, value=value)

    paged.free(seq_id=0)

    stats = paged.stats()

    # Reserved memory remains because the page store still exists.
    assert stats["live_kv_memory_bytes"] == 0
    assert stats["allocated_kv_memory_bytes"] == 0
    assert stats["reserved_cache_memory_bytes"] > 0
    assert stats["memory_used_bytes"] == 0