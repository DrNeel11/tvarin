"""Layer-level tier manager: the first concrete instance of the "AI Memory
Scheduler" from the architecture doc, minus the "AI" part -- Phase 3 is
static partitioning + synchronous (blocking) fetch-on-demand, which doubles
as the "naive remote offloading" baseline (strategy B in the Phase 6 plan).
Phase 4 adds prefetch-ahead on top of the same hook machinery.

Tier assignment is fixed at setup, matching the spec's own example:
    VRAM: layers 0..K-1 (never touched)
    RAM / Remote: layers K..N-1 (fetched on demand, evicted right after use)

This models a model that doesn't fit in VRAM: the overflow layers have no
permanent home, so on every single forward pass (i.e. every decode step)
they must be fetched again and evicted again to make room for the next one
-- exactly the pattern behind Phase 1's observed 7B CPU-offload collapse
(27 tok/s -> 1.1 tok/s), just now instrumented at the individual-layer
level instead of left to accelerate's opaque device_map dispatch.

Phase 5 adds one more mechanism: `cache_after_fetch`. Model weights are
static -- they never change between tokens -- so evicting a fetched layer
and re-fetching it over the network on every single decode step (Phase
3/4's behavior) throws away a fact a real scheduler should exploit: once
paid, a network fetch never needs to be paid again unless RAM itself is
under pressure. `cache_after_fetch=True` keeps fetched layers resident in
an LRU-bounded RAM cache instead of evicting them from RAM (only the
transient GPU copy is freed after each use), turning a recurring per-token
network cost into a one-time cost amortized over the whole generation.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import torch

from memtier.remote.client import RemoteMemoryClient
from memtier.tiered.layer_store import deserialize_layer
from memtier.tiered.shared_cache import SharedResidentCache

Tier = str  # "gpu" | "ram" | "remote"


@dataclass
class LayerOpRecord:
    layer_idx: int
    tier: Tier
    op: str  # "fetch" | "evict"
    wall_s: float
    fetch_wait_s: float | None = None
    fetch_transfer_s: float | None = None
    deserialize_s: float | None = None
    h2d_s: float | None = None
    prefetched: bool | None = None  # fetch only: was data already in-flight/ready when needed?
    ram_cache_hit: bool | None = None  # fetch only: served from the persistent RAM cache?


@dataclass
class _PrefetchState:
    event: threading.Event = field(default_factory=threading.Event)
    started_at: float = 0.0
    state: dict[str, torch.Tensor] | None = None
    fetch_transfer_s: float | None = None
    deserialize_s: float | None = None
    ram_cache_hit: bool = False


class LayerTierManager:
    def __init__(
        self,
        layers: list[torch.nn.Module],
        tier_map: dict[int, Tier],
        device: torch.device,
        remote_client: RemoteMemoryClient | None = None,
        run_id: str = "run",
        enable_prefetch: bool = False,
        # A second connection is needed for prefetching: the primary
        # remote_client's socket is used for the (possibly blocking)
        # foreground fetch, so a background prefetch thread needs its own
        # connection to actually run concurrently rather than queuing
        # behind it on the same socket.
        prefetch_remote_client: RemoteMemoryClient | None = None,
        # Phase 5: keep fetched layers resident in RAM instead of evicting
        # them, so a "remote" layer only pays the network cost once.
        cache_after_fetch: bool = False,
        ram_cache_budget_bytes: float | None = None,
        # When a layer's live module can't be trusted to hold real data
        # (e.g. accelerate's own multi-device dispatch represents CPU-
        # offloaded layers as meta-tensor placeholders backed by its own
        # hooks), this callable(idx) -> state_dict is used instead of
        # extracting from the module. Called lazily, one layer at a time,
        # during setup -- NOT pre-materialized for every layer up front,
        # which would double-buffer every layer's weights in memory at
        # once (state_loader's result + this manager's own copy of it)
        # right when many-GB CPU tiers are involved.
        state_loader=None,
        # Multi-tenant scheduling experiment (Phase 8): pass the SAME
        # SharedResidentCache instance to several LayerTierManagers (one
        # per model/tenant) to let them compete for one global budget,
        # instead of each getting its own private, non-competing cache.
        resident_cache: SharedResidentCache | None = None,
    ):
        self.layers = layers
        self.tier_map = dict(tier_map)  # static "home" tier for each layer index
        self.device = device
        self.remote_client = remote_client
        self.prefetch_remote_client = prefetch_remote_client or remote_client
        self.run_id = run_id
        self.enable_prefetch = enable_prefetch
        self.cache_after_fetch = cache_after_fetch
        self.ram_cache_budget_bytes = ram_cache_budget_bytes
        self._ram_cache: dict[int, dict[str, torch.Tensor]] = {}
        self._handles = []
        self.ops: list[LayerOpRecord] = []
        self._lock = threading.Lock()
        self._prefetches: dict[int, _PrefetchState] = {}
        self._overflow_layers = sorted(
            i for i, t in self.tier_map.items() if t in ("ram", "remote")
        )
        # Persistent RAM cache (Phase 5 only; unused when
        # cache_after_fetch=False). Private to this manager unless a
        # shared instance was passed in (Phase 8 multi-tenant case).
        self._resident_cache = resident_cache or SharedResidentCache(ram_cache_budget_bytes)
        self._state_loader = state_loader

        for idx, layer in enumerate(layers):
            home = self.tier_map.get(idx, "gpu")
            if home in ("ram", "remote"):
                self._offload_to_home(idx, layer, home)

    def _offload_to_home(self, idx: int, layer: torch.nn.Module, home: Tier):
        preloaded_state = self._state_loader(idx) if self._state_loader is not None else None
        if preloaded_state is not None:
            state = preloaded_state
        else:
            state = {name: p.detach().to("cpu").clone() for name, p in layer.named_parameters()}

        if home == "ram":
            self._ram_cache[idx] = state
        elif home == "remote":
            assert self.remote_client is not None, "remote tier requires a connected client"
            self.remote_client.put(f"{self.run_id}/layer_{idx}", _state_to_bytes(state))

        # fallback_device="cpu": a preloaded layer's live parameters may
        # still be meta tensors (accelerate's own offload representation);
        # torch.empty(0, device="meta") stays meta and can't be used, so
        # fall back to a real device for the placeholder in that case.
        _empty_layer_params(layer, fallback_device="cpu")
        torch.cuda.empty_cache()

    def install_hooks(self):
        for idx, layer in enumerate(self.layers):
            if self.tier_map.get(idx, "gpu") == "gpu":
                continue  # resident layers need no hook
            pre = layer.register_forward_pre_hook(self._make_pre_hook(idx))
            post = layer.register_forward_hook(self._make_post_hook(idx))
            self._handles.extend([pre, post])

    def prefetch_all_remote_layers(self):
        """Phase 8: kick off ONE background worker that sequentially warms
        every overflow layer not already cache-resident -- meant to be
        called while some *other* tenant is generating, so this tenant's
        whole working set is warm by the time its own turn starts. Unlike
        Phase 4's per-layer prefetch (compute window ~1-2ms, hopeless
        against a ~150ms transfer), the window available here is another
        tenant's entire generation (seconds), vastly larger than the time
        needed to prefetch a handful of layers -- the same mechanism,
        applied at a granularity where the timing argument works out.

        Sequential, not one thread per layer: prefetch_remote_client is a
        single persistent socket, and this protocol has no pipelining --
        concurrent threads issuing GETs on the same socket would interleave
        their request/response bytes and corrupt the stream.
        """
        def _worker():
            for idx in self._overflow_layers:
                if not self._resident_cache.contains(self._cache_key(idx)):
                    self._fetch_one(idx, self.prefetch_remote_client)

        threading.Thread(target=_worker, daemon=True).start()

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def _make_pre_hook(self, idx: int):
        def _hook(module, args, kwargs=None):
            self._fetch_to_gpu(idx, module)
            return None

        return _hook

    def _make_post_hook(self, idx: int):
        def _hook(module, args, output):
            self._evict_from_gpu(idx, module)
            return output

        return _hook

    def _layer_bytes(self, state: dict[str, torch.Tensor]) -> int:
        return sum(t.numel() * t.element_size() for t in state.values())

    def _cache_key(self, idx: int) -> str:
        return f"{self.run_id}:{idx}"

    def _ram_cache_get(self, idx: int) -> dict[str, torch.Tensor] | None:
        return self._resident_cache.get(self._cache_key(idx))

    def _ram_cache_put(self, idx: int, state: dict[str, torch.Tensor]):
        self._resident_cache.put(self._cache_key(idx), state, self._layer_bytes(state))

    def _fetch_one(self, idx: int, client: RemoteMemoryClient):
        """The network+deserialize part only -- shared by the foreground
        (cold) path and the background prefetch thread. Returns
        (state, transfer_s, deserialize_s, ram_cache_hit)."""
        if self.cache_after_fetch:
            cached = self._ram_cache_get(idx)
            if cached is not None:
                return cached, None, None, True

        home = self.tier_map[idx]
        if home == "ram":
            state, transfer_s, deser_s = self._ram_cache[idx], None, None
        elif home == "remote":
            r = client.get(f"{self.run_id}/layer_{idx}")
            assert r.found, f"layer {idx} missing from remote store"
            t_deser0 = time.perf_counter()
            state = deserialize_layer(r.payload)
            deser_s = time.perf_counter() - t_deser0
            transfer_s = r.total_s
        else:
            raise ValueError(home)

        if self.cache_after_fetch:
            self._ram_cache_put(idx, state)
        return state, transfer_s, deser_s, False

    def _start_prefetch(self, idx: int):
        if idx not in self._overflow_layers or idx in self._prefetches:
            return
        if self.cache_after_fetch and self._resident_cache.contains(self._cache_key(idx)):
            return  # already resident, nothing to prefetch
        pf = _PrefetchState(started_at=time.perf_counter())
        self._prefetches[idx] = pf

        def _worker():
            state, transfer_s, deser_s, cache_hit = self._fetch_one(idx, self.prefetch_remote_client)
            pf.state = state
            pf.fetch_transfer_s = transfer_s
            pf.deserialize_s = deser_s
            pf.ram_cache_hit = cache_hit
            pf.event.set()

        threading.Thread(target=_worker, daemon=True).start()

    def _next_overflow_layer(self, idx: int) -> int | None:
        pos = self._overflow_layers.index(idx)
        return self._overflow_layers[pos + 1] if pos + 1 < len(self._overflow_layers) else None

    def _fetch_to_gpu(self, idx: int, layer: torch.nn.Module):
        home = self.tier_map[idx]
        t0 = time.perf_counter()
        fetch_wait_s = fetch_transfer_s = deser_s = None
        prefetched = False
        ram_cache_hit = False

        # Consumed whenever a prefetch is in flight/ready, regardless of
        # enable_prefetch: that flag only controls whether the per-layer
        # hook *triggers new* prefetches automatically; a prefetch started
        # some other way (e.g. prefetch_all_remote_layers(), Phase 8's
        # tenant-switch-aware prefetch) should still be picked up here.
        if idx in self._prefetches:
            pf = self._prefetches.pop(idx)
            t_wait0 = time.perf_counter()
            pf.event.wait()
            fetch_wait_s = time.perf_counter() - t_wait0
            state = pf.state
            fetch_transfer_s = pf.fetch_transfer_s
            deser_s = pf.deserialize_s
            ram_cache_hit = pf.ram_cache_hit
            prefetched = True
        else:
            state, fetch_transfer_s, deser_s, ram_cache_hit = self._fetch_one(idx, self.remote_client)
            fetch_wait_s = time.perf_counter() - t0

        t_h2d0 = time.perf_counter()
        for name, p in layer.named_parameters():
            p.data = state[name].to(self.device, non_blocking=False)
        torch.cuda.synchronize()
        h2d_s = time.perf_counter() - t_h2d0

        if self.enable_prefetch:
            nxt = self._next_overflow_layer(idx)
            if nxt is not None:
                self._start_prefetch(nxt)

        with self._lock:
            self.ops.append(LayerOpRecord(
                layer_idx=idx, tier=home, op="fetch", wall_s=time.perf_counter() - t0,
                fetch_wait_s=fetch_wait_s, fetch_transfer_s=fetch_transfer_s,
                deserialize_s=deser_s, h2d_s=h2d_s, prefetched=prefetched,
                ram_cache_hit=ram_cache_hit,
            ))

    def _evict_from_gpu(self, idx: int, layer: torch.nn.Module):
        t0 = time.perf_counter()
        _empty_layer_params(layer)
        with self._lock:
            self.ops.append(LayerOpRecord(
                layer_idx=idx, tier=self.tier_map[idx], op="evict",
                wall_s=time.perf_counter() - t0,
            ))


def _empty_layer_params(layer: torch.nn.Module, fallback_device: str | None = None):
    for p in layer.parameters():
        device = fallback_device if p.device.type == "meta" else p.device
        p.data = torch.empty(0, device=device, dtype=p.dtype)


def _state_to_bytes(state: dict[str, torch.Tensor]) -> bytes:
    import io
    buf = io.BytesIO()
    torch.save(state, buf)
    return buf.getvalue()
