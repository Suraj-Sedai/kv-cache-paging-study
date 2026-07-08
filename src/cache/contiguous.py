from typing import Union

import torch
from src.cache.base import BaseKVCache


class ContiguousKVCache(BaseKVCache):
    """
    Baseline KV cache using preallocated contiguous tensors.

    Layout:
        keys[layer, seq, kv_head, token_pos, head_dim]
        values[layer, seq, kv_head, token_pos, head_dim]

    This baseline is intentionally simple. Its read path returns views,
    not copied tensors.
    """

    def __init__(
        self,
        num_layers: int,
        max_batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
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

        self.num_layers = num_layers
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)

        shape = (
            num_layers,
            max_batch_size,
            num_kv_heads,
            max_seq_len,
            head_dim,
        )

        self.keys = torch.empty(shape, dtype=dtype, device=self.device)
        self.values = torch.empty(shape, dtype=dtype, device=self.device)

        self.lengths = torch.zeros(max_batch_size, dtype=torch.long, device="cpu")
        self.active = torch.zeros(max_batch_size, dtype=torch.bool, device="cpu")

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

        self.keys[layer_id, seq_id, :, token_pos, :] = key
        self.values[layer_id, seq_id, :, token_pos, :] = value

        self.active[seq_id] = True
        self.lengths[seq_id] = max(int(self.lengths[seq_id]), token_pos + 1)

    def read(self, layer_id: int, seq_id: int, upto_pos: int):
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

        key_view = self.keys[layer_id, seq_id, :, :upto_pos, :]
        value_view = self.values[layer_id, seq_id, :, :upto_pos, :]

        return key_view, value_view

    def advance(self, seq_id: int) -> None:
        self._validate_seq(seq_id)

        if not bool(self.active[seq_id]):
            raise KeyError(f"seq_id {seq_id} is not active")

        # The contiguous cache does not need separate decode-position state yet.
        # This method exists to preserve the shared cache interface.
        return None

    def free(self, seq_id: int) -> None:
        self._validate_seq(seq_id)

        self.active[seq_id] = False
        self.lengths[seq_id] = 0

    def memory_used(self) -> int:
        return self.allocated_kv_memory_bytes()

    def fragmentation(self) -> dict:
        used_tokens = int(self.lengths[self.active].sum().item())
        active_sequences = int(self.active.sum().item())

        allocated_tokens = active_sequences * self.max_seq_len
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
            "cache_type": "contiguous",
            "num_layers": self.num_layers,
            "max_batch_size": self.max_batch_size,
            "max_seq_len": self.max_seq_len,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "active_sequences": int(self.active.sum().item()),
            "memory_used_bytes": self.memory_used(),
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
        active_sequences = int(self.active.sum().item())
        allocated_tokens = active_sequences * self.max_seq_len
        return allocated_tokens * self._per_token_kv_bytes()


    def reserved_cache_memory_bytes(self) -> int:
        reserved_tokens = self.max_batch_size * self.max_seq_len
        return reserved_tokens * self._per_token_kv_bytes()