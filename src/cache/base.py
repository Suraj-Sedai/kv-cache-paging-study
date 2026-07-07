from abc import ABC, abstractmethod


class BaseKVCache(ABC):
    """
    Common interface for all KV-cache implementations.

    Every cache layout must implement this API so that experiments compare
    cache behavior under the same runner and decode loop.
    """

    @abstractmethod
    def write(self, layer_id: int, seq_id: int, token_pos: int, key, value) -> None:
        """
        Store key/value tensors for one token position.
        """
        raise NotImplementedError

    @abstractmethod
    def read(self, layer_id: int, seq_id: int, upto_pos: int):
        """
        Return key/value tensors for positions [0, upto_pos).

        Implementations must not return unwritten padded tokens.
        """
        raise NotImplementedError

    @abstractmethod
    def advance(self, seq_id: int) -> None:
        """
        Advance sequence state by one decode step if the implementation tracks it.
        """
        raise NotImplementedError

    @abstractmethod
    def free(self, seq_id: int) -> None:
        """
        Release all cache memory associated with a finished sequence.
        """
        raise NotImplementedError

    @abstractmethod
    def memory_used(self) -> int:
        """
        Return implementation-level cache memory usage in bytes.
        """
        raise NotImplementedError

    @abstractmethod
    def fragmentation(self) -> dict:
        """
        Return fragmentation statistics.

        Expected keys should eventually include:
        used_tokens, allocated_tokens, wasted_tokens, fragmentation_ratio.
        """
        raise NotImplementedError

    @abstractmethod
    def stats(self) -> dict:
        """
        Return implementation-specific counters useful for benchmarking.
        """
        raise NotImplementedError