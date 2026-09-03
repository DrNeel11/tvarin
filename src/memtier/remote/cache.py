"""Bounded local LRU cache for blocks fetched from the remote memory tier.

Kept separate from RemoteMemoryClient (rather than baked in) so Phase 3's
tier-aware scheduler can swap in its own eviction policy without touching
the network code.
"""

from __future__ import annotations

from collections import OrderedDict


class LRUBlockCache:
    def __init__(self, capacity_bytes: int):
        self.capacity_bytes = capacity_bytes
        self._data: OrderedDict[str, bytes] = OrderedDict()
        self._size_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: str) -> bytes | None:
        if key in self._data:
            self._data.move_to_end(key)
            self.hits += 1
            return self._data[key]
        self.misses += 1
        return None

    def put(self, key: str, value: bytes):
        if key in self._data:
            self._size_bytes -= len(self._data[key])
            del self._data[key]
        self._data[key] = value
        self._size_bytes += len(value)
        self._data.move_to_end(key)
        while self._size_bytes > self.capacity_bytes and self._data:
            evicted_key, evicted_val = self._data.popitem(last=False)
            self._size_bytes -= len(evicted_val)
            self.evictions += 1

    def contains(self, key: str) -> bool:
        return key in self._data

    @property
    def hit_rate(self) -> float | None:
        total = self.hits + self.misses
        return self.hits / total if total else None
