"""A byte-budgeted LRU cache that can be shared across multiple
LayerTierManager instances (i.e. multiple models/tenants), for the
multi-tenant scheduling experiment: does letting several tenants compete
for one global cache (adaptive, favors whoever's active) beat giving each
a fixed, non-competing partition of the same total budget?

Keys are caller-supplied and expected to be namespaced (e.g.
"{run_id}:{layer_idx}") so multiple tenants sharing one instance don't
collide.
"""

from __future__ import annotations

import threading
from collections import OrderedDict


class SharedResidentCache:
    def __init__(self, budget_bytes: float | None):
        self.budget_bytes = budget_bytes
        self._data: OrderedDict[str, tuple[dict, int]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        # eviction log entries: (evicted_key, evicting_key) -- lets a
        # multi-tenant experiment see *whose* data got evicted by whom.
        self.eviction_log: list[tuple[str, str]] = []

    def get(self, key: str):
        with self._lock:
            entry = self._data.get(key)
            if entry is not None:
                self._data.move_to_end(key)
                self.hits += 1
                return entry[0]
            self.misses += 1
            return None

    def put(self, key: str, state: dict, size_bytes: int):
        with self._lock:
            if key in self._data:
                self._bytes -= self._data[key][1]
            self._data[key] = (state, size_bytes)
            self._data.move_to_end(key)
            self._bytes += size_bytes
            if self.budget_bytes is not None:
                while self._bytes > self.budget_bytes and len(self._data) > 1:
                    evict_key, (_, evict_size) = self._data.popitem(last=False)
                    self._bytes -= evict_size
                    self.evictions += 1
                    self.eviction_log.append((evict_key, key))

    def contains(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    @property
    def hit_rate(self) -> float | None:
        total = self.hits + self.misses
        return self.hits / total if total else None
