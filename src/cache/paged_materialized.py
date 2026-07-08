from typing import Dict, List, Tuple, Union
import math

import torch

from src.cache.base import BaseKVCache


class PagedMaterializedKVCache(BaseKVCache):
    """
    Paged KV cache using fixed-size blocks/pages.

    Layout:
        keys[layer, physical_page, kv_head, offset_within_page, head_dim]
        values[layer, physical_page, kv_head, offset_within_page, head_dim]

    Each sequence owns a block table:
        seq_id -> [physical_page_id_for_logical_page_0, ...]

    Important:
        read(...) materializes scattered pages into a contiguous tensor.
        This is intentionally not native PagedAttention.
    """

    def __init__(
        self,
        num_layers: int,
        max_batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        dtype=torch.float32,
        device: Union[str, torch.device] = "cpu",
    ):
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")

        self.num_layers = num_layers
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.dtype = dtype
        self.device = torch.device(device)

        self.max_pages_per_sequence = int(math.ceil(max_seq_len / block_size))
        self.max_pages = max_batch_size * self.max_pages_per_sequence

        page_shape = (
            num_layers,
            self.max_pages,
            num_kv_heads,
            block_size,
            head_dim,
        )

        self.keys = torch.empty(page_shape, dtype=dtype, device=self.device)
        self.values = torch.empty(page_shape, dtype=dtype, device=self.device)

        self.lengths = torch.zeros(max_batch_size, dtype=torch.long, device="cpu")
        self.active = torch.zeros(max_batch_size, dtype=torch.bool, device="cpu")

        self.block_tables: Dict[int, List[int]] = {
            seq_id: [] for seq_id in range(max_batch_size)
        }

        # Stack of reusable physical page IDs.
        self.free_page_ids: List[int] = list(range(self.max_pages - 1, -1, -1))

        self.page_lookups = 0
        self.materialized_reads = 0
        self.copy_bytes = 0

    def _validate_layer(self, layer_id: int) -> None:
        if not 0 <= layer_id < self.num_layers:
            raise IndexError(f"layer_id {layer_id} out of range")

    def _validate_seq(self, seq_id: int) -> None:
        if not 0 <= seq_id < self.max_batch_size:
            raise IndexError(f"seq_id {seq_id} out of range")

    def _validate_key_value(self, key: torch.Tensor, value: torch.Tensor) -> None:
        expected_shape = (self.num_kv_heads, self.head_dim)

        if key.shape != expected_shape:
            raise ValueError(f"key shape must be {expected_shape}, got {tuple(key.shape)}")

        if value.shape != expected_shape:
            raise ValueError(f"value shape must be {expected_shape}, got {tuple(value.shape)}")

        if key.dtype != self.dtype:
            raise TypeError(f"key dtype must be {self.dtype}, got {key.dtype}")

        if value.dtype != self.dtype:
            raise TypeError(f"value dtype must be {self.dtype}, got {value.dtype}")

    def _allocate_page(self) -> int:
        if not self.free_page_ids:
            raise RuntimeError("PagedMaterializedKVCache is out of free pages")

        return self.free_page_ids.pop()

    def _ensure_logical_page(self, seq_id: int, logical_page_idx: int) -> int:
        table = self.block_tables[seq_id]

        if logical_page_idx >= self.max_pages_per_sequence:
            raise IndexError(
                f"logical_page_idx {logical_page_idx} exceeds max pages per sequence"
            )

        while len(table) <= logical_page_idx:
            table.append(self._allocate_page())

        return table[logical_page_idx]

    def write(self, layer_id: int, seq_id: int, token_pos: int, key, value) -> None:
        self._validate_layer(layer_id)
        self._validate_seq(seq_id)

        if token_pos < 0 or token_pos >= self.max_seq_len:
            raise IndexError(f"token_pos {token_pos} out of range")

        if not isinstance(key, torch.Tensor):
            key = torch.as_tensor(key, dtype=self.dtype, device=self.device)
        else:
            key = key.to(device=self.device)

        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        else:
            value = value.to(device=self.device)

        self._validate_key_value(key, value)

        logical_page_idx = token_pos // self.block_size
        offset = token_pos % self.block_size

        physical_page_id = self._ensure_logical_page(seq_id, logical_page_idx)

        self.keys[layer_id, physical_page_id, :, offset, :] = key
        self.values[layer_id, physical_page_id, :, offset, :] = value

        self.active[seq_id] = True
        self.lengths[seq_id] = max(int(self.lengths[seq_id]), token_pos + 1)

    def read(self, layer_id: int, seq_id: int, upto_pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._validate_layer(layer_id)
        self._validate_seq(seq_id)

        if not bool(self.active[seq_id]):
            raise KeyError(f"seq_id {seq_id} is not active")

        current_len = int(self.lengths[seq_id])

        if upto_pos < 0:
            raise ValueError("upto_pos must be non-negative")

        if upto_pos > current_len:
            raise ValueError(
                f"cannot read upto_pos={upto_pos}; only {current_len} tokens written"
            )

        out_key = torch.empty(
            (self.num_kv_heads, upto_pos, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        out_value = torch.empty(
            (self.num_kv_heads, upto_pos, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )

        table = self.block_tables[seq_id]

        read_pos = 0
        while read_pos < upto_pos:
            logical_page_idx = read_pos // self.block_size
            offset = read_pos % self.block_size
            remaining_in_page = self.block_size - offset
            remaining_total = upto_pos - read_pos
            chunk = min(remaining_in_page, remaining_total)

            physical_page_id = table[logical_page_idx]
            self.page_lookups += 1

            out_key[:, read_pos : read_pos + chunk, :] = self.keys[
                layer_id,
                physical_page_id,
                :,
                offset : offset + chunk,
                :,
            ]
            out_value[:, read_pos : read_pos + chunk, :] = self.values[
                layer_id,
                physical_page_id,
                :,
                offset : offset + chunk,
                :,
            ]

            copied_elements = 2 * self.num_kv_heads * chunk * self.head_dim
            self.copy_bytes += copied_elements * self._element_size_bytes()

            read_pos += chunk

        self.materialized_reads += 1

        return out_key, out_value

    def advance(self, seq_id: int) -> None:
        self._validate_seq(seq_id)

        if not bool(self.active[seq_id]):
            raise KeyError(f"seq_id {seq_id} is not active")

        # Decode-position tracking is not needed yet.
        # The method exists to preserve the shared cache interface.
        return None

    def free(self, seq_id: int) -> None:
        self._validate_seq(seq_id)

        table = self.block_tables[seq_id]

        self.free_page_ids.extend(table)
        self.block_tables[seq_id] = []

        self.active[seq_id] = False
        self.lengths[seq_id] = 0

    def memory_used(self) -> int:
        return self.allocated_kv_memory_bytes()
    
    def _allocated_page_count(self) -> int:
        return self.max_pages - len(self.free_page_ids)

    def fragmentation(self) -> dict:
        used_tokens = int(self.lengths[self.active].sum().item())
        allocated_pages = self._allocated_page_count()
        allocated_tokens = allocated_pages * self.block_size
        wasted_tokens = allocated_tokens - used_tokens

        if allocated_tokens == 0:
            fragmentation_ratio = 0.0
        else:
            fragmentation_ratio = wasted_tokens / allocated_tokens

        return {
            "used_tokens": used_tokens,
            "allocated_tokens": allocated_tokens,
            "wasted_tokens": wasted_tokens,
            "fragmentation_ratio": fragmentation_ratio,
        }

    def stats(self) -> dict:
        return {
            "cache_type": "paged_materialized",
            "num_layers": self.num_layers,
            "max_batch_size": self.max_batch_size,
            "max_seq_len": self.max_seq_len,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "block_size": self.block_size,
            "max_pages": self.max_pages,
            "allocated_pages": self._allocated_page_count(),
            "free_pages": len(self.free_page_ids),
            "active_sequences": int(self.active.sum().item()),
            "memory_used_bytes": self.memory_used(),
            "page_lookups": self.page_lookups,
            "materialized_reads": self.materialized_reads,
            "copy_bytes": self.copy_bytes,
            "live_kv_memory_bytes": self.live_kv_memory_bytes(),
            "allocated_kv_memory_bytes": self.allocated_kv_memory_bytes(),
            "reserved_cache_memory_bytes": self.reserved_cache_memory_bytes(),
        }
    def _element_size_bytes(self) -> int:
        return torch.tensor([], dtype=self.dtype).element_size()


    def _per_token_kv_bytes(self) -> int:
        return (
            self.num_layers
            * 2
            * self.num_kv_heads
            * self.head_dim
            * self._element_size_bytes()
        )


    def live_kv_memory_bytes(self) -> int:
        used_tokens = int(self.lengths[self.active].sum().item())
        return used_tokens * self._per_token_kv_bytes()


    def allocated_kv_memory_bytes(self) -> int:
        allocated_tokens = self._allocated_page_count() * self.block_size
        return allocated_tokens * self._per_token_kv_bytes()


    def reserved_cache_memory_bytes(self) -> int:
        reserved_tokens = self.max_pages * self.block_size
        return reserved_tokens * self._per_token_kv_bytes()