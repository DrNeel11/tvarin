"""Phase 8: multi-tenant scheduling experiment.

Follows directly from Phase 7's reframed research question: not "can one
user's remote-memory access be made fast" (Phases 3-4 answered that: no,
not via prefetching) but "can intelligent scheduling make remote memory
practical across *multiple concurrent* workloads." Two real models
(Qwen2.5-0.5B and -1.5B) are loaded simultaneously, each with some layers
forced onto a shared "remote" tier, and served with interleaved requests
(A, B, A, B, ...) -- simulating two tenants competing for the same
VRAM+RAM+remote budget on one machine.

Compares three RAM-cache allocation/scheduling policies at the same total
budget:
  - "shared": one global cache both tenants' fetched layers compete for,
    reactively (a layer is only fetched when that tenant's own forward
    pass needs it) -- adaptive, at the cost of a cold-refetch at the start
    of each tenant's turn if the other tenant evicted its data meanwhile.
  - "partitioned": each tenant gets a fixed, non-competing half of the
    same total budget (simple, predictable, no cross-tenant eviction --
    but wastes budget if one tenant needs more than its fixed share).
  - "shared_prefetch": same shared pool, but tenant-switch-aware: while
    tenant X is generating (its own ~128-token turn, taking seconds), the
    *other* tenant's full working set is prefetched in the background so
    it's already warm by the time its turn starts. Unlike Phase 4's
    per-layer prefetch (compute window ~1-2ms vs. a ~150ms transfer, no
    contest), the window here is another tenant's entire turn -- vastly
    larger than the time needed to prefetch a handful of layers.

Usage:
    python scripts/run_multi_tenant.py --scheme shared
    python scripts/run_multi_tenant.py --scheme partitioned
    python scripts/run_multi_tenant.py --scheme shared_prefetch
"""

import argparse
import csv
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import memtier.config  # noqa: F401
import torch

from memtier.benchmark.harness import load_model, run_one_prompt
from memtier.config import BENCHMARK_PROMPTS, RESULTS_DIR
from memtier.models.registry import get_by_label
from memtier.remote.client import RemoteMemoryClient
from memtier.remote.server import RemoteMemoryServer
from memtier.tiered.shared_cache import SharedResidentCache
from memtier.tiered.tier_manager import LayerTierManager

PROMPT_TEXT = list(BENCHMARK_PROMPTS.values())[0]  # "short" -- keep rounds fast
N_ROUNDS = 6


def setup_tenant(model_label, overflow_layers, run_id, remote_client):
    cfg = get_by_label(model_label)
    model, tokenizer, load_time_s, summary = load_model(cfg)
    n_layers = len(model.model.layers)
    n_gpu = n_layers - overflow_layers
    tier_map = {i: "gpu" for i in range(n_gpu)}
    tier_map.update({i: "remote" for i in range(n_gpu, n_layers)})
    print(f"[{run_id}] {model_label}: {n_layers} layers, {n_gpu} gpu-resident, "
          f"{overflow_layers} remote (loaded in {load_time_s:.1f}s, {summary})")
    return model, tokenizer, tier_map, n_layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", choices=["shared", "partitioned", "shared_prefetch"], required=True)
    parser.add_argument("--latency-ms", type=float, default=10.0)
    parser.add_argument("--bandwidth-mb-per-s", type=float, default=1000.0)
    parser.add_argument("--overflow-a", type=int, default=6, help="0.5B overflow layer count")
    parser.add_argument("--overflow-b", type=int, default=8, help="1.5B overflow layer count")
    parser.add_argument("--total-budget-mb", type=float, default=750.0,
                         help="combined RAM cache budget across both tenants. Measured: A (0.5B, "
                              "6 layers) needs ~179MB, B (1.5B, 8 layers) needs ~749MB fully "
                              "cached; default ~= B's need alone, comfortably above A's, well "
                              "below the ~928MB combined -- sized so the two policies diverge "
                              "instead of both collapsing to the same undersized-cache cliff")
    parser.add_argument("--tag", default="phase8_multi_tenant")
    args = parser.parse_args()

    server = RemoteMemoryServer(
        host="127.0.0.1", port=5980, latency_ms=args.latency_ms,
        bandwidth_mb_per_s=args.bandwidth_mb_per_s,
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(0.3)

    client_a = RemoteMemoryClient(host="127.0.0.1", port=5980, upload_bandwidth_mb_per_s=args.bandwidth_mb_per_s)
    client_a.connect()
    client_b = RemoteMemoryClient(host="127.0.0.1", port=5980, upload_bandwidth_mb_per_s=args.bandwidth_mb_per_s)
    client_b.connect()
    # Dedicated connections for background prefetch -- must not share a
    # socket with the foreground client, since our protocol reads one
    # response per request in order; a prefetch thread and the main thread
    # issuing requests on the same socket would corrupt that ordering.
    prefetch_client_a = RemoteMemoryClient(host="127.0.0.1", port=5980, upload_bandwidth_mb_per_s=args.bandwidth_mb_per_s)
    prefetch_client_a.connect()
    prefetch_client_b = RemoteMemoryClient(host="127.0.0.1", port=5980, upload_bandwidth_mb_per_s=args.bandwidth_mb_per_s)
    prefetch_client_b.connect()

    model_a, tok_a, tier_map_a, n_layers_a = setup_tenant(
        "qwen2.5-0.5b-fp16", args.overflow_a, "tenantA", client_a)
    model_b, tok_b, tier_map_b, n_layers_b = setup_tenant(
        "qwen2.5-1.5b-fp16", args.overflow_b, "tenantB", client_b)

    total_budget_bytes = args.total_budget_mb * 1_000_000
    if args.scheme in ("shared", "shared_prefetch"):
        shared = SharedResidentCache(budget_bytes=total_budget_bytes)
        cache_kwargs_a = dict(resident_cache=shared)
        cache_kwargs_b = dict(resident_cache=shared)
    else:
        cache_kwargs_a = dict(ram_cache_budget_bytes=total_budget_bytes / 2)
        cache_kwargs_b = dict(ram_cache_budget_bytes=total_budget_bytes / 2)

    device = torch.device("cuda:0")
    manager_a = LayerTierManager(
        model_a.model.layers, tier_map_a, device=device, remote_client=client_a,
        prefetch_remote_client=prefetch_client_a,
        run_id="tenantA", cache_after_fetch=True, **cache_kwargs_a,
    )
    manager_b = LayerTierManager(
        model_b.model.layers, tier_map_b, device=device, remote_client=client_b,
        prefetch_remote_client=prefetch_client_b,
        run_id="tenantB", cache_after_fetch=True, **cache_kwargs_b,
    )
    manager_a.install_hooks()
    manager_b.install_hooks()

    print(f"\n=== scheme={args.scheme}, total_budget={args.total_budget_mb}MB, "
          f"interleaving {N_ROUNDS} rounds of (A, B) ===\n")

    round_results = []
    for round_idx in range(N_ROUNDS):
        if args.scheme == "shared_prefetch":
            manager_b.prefetch_all_remote_layers()  # warm B while A runs below
        ra = run_one_prompt(model_a, tok_a, "short", PROMPT_TEXT)

        if args.scheme == "shared_prefetch":
            manager_a.prefetch_all_remote_layers()  # warm A (for next round) while B runs below
        rb = run_one_prompt(model_b, tok_b, "short", PROMPT_TEXT)

        round_results.append(dict(
            round=round_idx,
            tenant_a_decode_tps=ra["decode_tokens_per_s"],
            tenant_b_decode_tps=rb["decode_tokens_per_s"],
        ))
        print(f"round {round_idx}: A={ra['decode_tokens_per_s']:.2f}tok/s  "
              f"B={rb['decode_tokens_per_s']:.2f}tok/s")

    def cache_stats(cache: SharedResidentCache, label: str):
        cross = sum(
            1 for evicted_key, evicting_key in cache.eviction_log
            if evicted_key.split(":")[0] != evicting_key.split(":")[0]
        )
        print(f"[{label}] hits={cache.hits} misses={cache.misses} "
              f"hit_rate={cache.hit_rate:.3f} evictions={cache.evictions} "
              f"cross_tenant_evictions={cross}")
        return dict(hits=cache.hits, misses=cache.misses, hit_rate=cache.hit_rate,
                     evictions=cache.evictions, cross_tenant_evictions=cross)

    print()
    if args.scheme in ("shared", "shared_prefetch"):
        cache_summary = {"shared": cache_stats(shared, "shared")}
    else:
        cache_summary = {
            "tenantA": cache_stats(manager_a._resident_cache, "tenantA"),
            "tenantB": cache_stats(manager_b._resident_cache, "tenantB"),
        }

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / f"{ts}_{args.tag}_{args.scheme}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "rounds.json", "w") as f:
        json.dump(round_results, f, indent=2)
    with open(out_dir / "cache_summary.json", "w") as f:
        json.dump(cache_summary, f, indent=2)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args) | dict(n_layers_a=n_layers_a, n_layers_b=n_layers_b), f, indent=2)
    print(f"\nResults saved to {out_dir}")

    manager_a.remove_hooks()
    manager_b.remove_hooks()
    server.stop()


if __name__ == "__main__":
    main()
