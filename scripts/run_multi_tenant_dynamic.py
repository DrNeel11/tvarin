"""Phase 10: does prediction quality matter once the workload isn't a
clean, known alternation?

Phase 8's coda showed tenant-switch prefetch working -- but it assumed the
schedule (A, B, A, B, ...) and always prefetched "the other tenant." That
assumption was true by construction there. Real workloads aren't strictly
alternating. This tests three policies against a workload with bursts
(steady alternation, then a burst of only A, then a burst of only B, then
back to alternation) -- the regime changes are exactly where a
fixed-pattern assumption should break and an adaptive one should recover:

  - "reactive": shared LRU cache, no prefetch (Phase 8's first baseline).
  - "naive": always prefetch "the other tenant" next, regardless of actual
    history (Phase 8 coda's exact policy, unchanged) -- expected to
    actively hurt during bursts, since it evicts the *correct* tenant's
    warm cache to make room for a prefetch that turns out to be wrong.
  - "adaptive": a first-order Markov transition model, learned online from
    observed (prev_tenant -> next_tenant) history, predicts whichever
    tenant most often followed the current one so far and prefetches that
    -- the "start with a lightweight model" guidance applied literally: an
    empirical transition-frequency table, not a trained neural net.

Usage:
    python scripts/run_multi_tenant_dynamic.py --scheme reactive
    python scripts/run_multi_tenant_dynamic.py --scheme naive
    python scripts/run_multi_tenant_dynamic.py --scheme adaptive
"""

import argparse
import json
import sys
import threading
import time
from collections import defaultdict
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

PROMPT_TEXT = list(BENCHMARK_PROMPTS.values())[0]

# Steady alternation -> burst of A -> burst of B -> steady alternation.
# The bursts are where a fixed "always assume alternation" prefetch policy
# should actively mispredict, and where an adaptive one has to catch up.
SCHEDULE = (
    ["A", "B"] * 2 +      # rounds 0-3: steady
    ["A"] * 4 +            # rounds 4-7: burst of A
    ["B"] * 4 +            # rounds 8-11: burst of B
    ["A", "B"] * 2          # rounds 12-15: back to steady
)


class MarkovTenantPredictor:
    """Learned online: for each tenant, which tenant most often ran next --
    within a *recency window*, not unbounded cumulative counts. Unbounded
    counting was tried first and failed silently: 2 early A->B observations
    from a steady-alternation phase took 3+ A->A observations to outweigh,
    so the predictor never flipped within a 4-round burst -- identical
    behavior to the naive fixed-pattern policy, for a different reason.
    A model that weighs ancient history as heavily as the last few rounds
    isn't actually adaptive; recency weighting is what "adaptive" needs to
    mean here."""

    def __init__(self, tenants: list[str], window: int = 3):
        self.tenants = tenants
        self.window = window
        self.history: dict[str, list[str]] = {t: [] for t in tenants}
        self._last: str | None = None

    def observe(self, tenant: str):
        if self._last is not None:
            h = self.history[self._last]
            h.append(tenant)
            if len(h) > self.window:
                h.pop(0)
        self._last = tenant

    def predict_next(self, current: str) -> str:
        recent = self.history[current]
        if not recent:
            others = [t for t in self.tenants if t != current]
            return others[0] if others else current
        counts = defaultdict(int)
        for t in recent:
            counts[t] += 1
        return max(counts, key=counts.get)


class ConfidenceGatedPredictor:
    """Phase 10 follow-up. window=1 adaptive beat naive but still lost to
    reactive: it acts on every single observation, including the one right
    after a regime boundary where the "recent" evidence is stale and about
    to be wrong. This only prefetches when the last `confidence_threshold`
    observations from the current tenant all agree -- otherwise it does
    nothing (falls back to reactive for that round), trading fewer
    opportunities to act for fewer confidently-wrong, budget-costly bets."""

    def __init__(self, tenants: list[str], confidence_threshold: int = 2):
        self.tenants = tenants
        self.confidence_threshold = confidence_threshold
        self.history: dict[str, list[str]] = {t: [] for t in tenants}
        self._last: str | None = None

    def observe(self, tenant: str):
        if self._last is not None:
            h = self.history[self._last]
            h.append(tenant)
            if len(h) > self.confidence_threshold:
                h.pop(0)
        self._last = tenant

    def predict_next_if_confident(self, current: str) -> str | None:
        recent = self.history[current]
        if len(recent) < self.confidence_threshold:
            return None  # not enough evidence yet
        if len(set(recent)) == 1:
            return recent[0]  # unanimous over the window
        return None  # disagreement -- don't act on ambiguous evidence


def setup_tenant(model_label, overflow_layers, run_id, remote_client, prefetch_client):
    cfg = get_by_label(model_label)
    model, tokenizer, load_time_s, summary = load_model(cfg)
    n_layers = len(model.model.layers)
    n_gpu = n_layers - overflow_layers
    tier_map = {i: "gpu" for i in range(n_gpu)}
    tier_map.update({i: "remote" for i in range(n_gpu, n_layers)})
    print(f"[{run_id}] {model_label}: {n_layers} layers, {n_gpu} gpu-resident, "
          f"{overflow_layers} remote (loaded in {load_time_s:.1f}s, {summary})")
    return model, tokenizer, n_layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", choices=["reactive", "naive", "adaptive", "confident"], required=True)
    parser.add_argument("--confidence-threshold", type=int, default=2,
                         help="confident scheme only: consecutive agreeing observations required to act")
    parser.add_argument("--latency-ms", type=float, default=10.0)
    parser.add_argument("--bandwidth-mb-per-s", type=float, default=1000.0)
    parser.add_argument("--overflow-a", type=int, default=6)
    parser.add_argument("--overflow-b", type=int, default=8)
    parser.add_argument("--total-budget-mb", type=float, default=750.0)
    parser.add_argument("--tag", default="phase10_dynamic")
    args = parser.parse_args()

    server = RemoteMemoryServer(host="127.0.0.1", port=5990, latency_ms=args.latency_ms,
                                 bandwidth_mb_per_s=args.bandwidth_mb_per_s)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(0.3)

    clients = {}
    prefetch_clients = {}
    for name in ("A", "B"):
        c = RemoteMemoryClient(host="127.0.0.1", port=5990, upload_bandwidth_mb_per_s=args.bandwidth_mb_per_s)
        c.connect()
        clients[name] = c
        pc = RemoteMemoryClient(host="127.0.0.1", port=5990, upload_bandwidth_mb_per_s=args.bandwidth_mb_per_s)
        pc.connect()
        prefetch_clients[name] = pc

    model_a, tok_a, n_layers_a = setup_tenant("qwen2.5-0.5b-fp16", args.overflow_a, "A", clients["A"], prefetch_clients["A"])
    model_b, tok_b, n_layers_b = setup_tenant("qwen2.5-1.5b-fp16", args.overflow_b, "B", clients["B"], prefetch_clients["B"])
    models = {"A": model_a, "B": model_b}
    toks = {"A": tok_a, "B": tok_b}

    shared = SharedResidentCache(budget_bytes=args.total_budget_mb * 1_000_000)
    device = torch.device("cuda:0")
    n_gpu_a = n_layers_a - args.overflow_a
    n_gpu_b = n_layers_b - args.overflow_b
    tier_map_a = {i: ("gpu" if i < n_gpu_a else "remote") for i in range(n_layers_a)}
    tier_map_b = {i: ("gpu" if i < n_gpu_b else "remote") for i in range(n_layers_b)}

    manager_a = LayerTierManager(model_a.model.layers, tier_map_a, device=device,
                                  remote_client=clients["A"], prefetch_remote_client=prefetch_clients["A"],
                                  run_id="A", cache_after_fetch=True, resident_cache=shared)
    manager_b = LayerTierManager(model_b.model.layers, tier_map_b, device=device,
                                  remote_client=clients["B"], prefetch_remote_client=prefetch_clients["B"],
                                  run_id="B", cache_after_fetch=True, resident_cache=shared)
    managers = {"A": manager_a, "B": manager_b}
    manager_a.install_hooks()
    manager_b.install_hooks()

    predictor = MarkovTenantPredictor(["A", "B"], window=1)
    confident_predictor = ConfidenceGatedPredictor(["A", "B"], confidence_threshold=args.confidence_threshold)

    print(f"\n=== scheme={args.scheme}, schedule={''.join(SCHEDULE)} ===\n")

    round_results = []
    prefetches_triggered = 0
    for round_idx, tenant in enumerate(SCHEDULE):
        if args.scheme == "naive":
            other = "B" if tenant == "A" else "A"
            managers[other].prefetch_all_remote_layers()
            prefetches_triggered += 1
        elif args.scheme == "adaptive":
            predicted = predictor.predict_next(tenant)
            if predicted != tenant:
                managers[predicted].prefetch_all_remote_layers()
                prefetches_triggered += 1
        elif args.scheme == "confident":
            predicted = confident_predictor.predict_next_if_confident(tenant)
            if predicted is not None and predicted != tenant:
                managers[predicted].prefetch_all_remote_layers()
                prefetches_triggered += 1

        r = run_one_prompt(models[tenant], toks[tenant], "short", PROMPT_TEXT)
        predictor.observe(tenant)
        confident_predictor.observe(tenant)

        round_results.append(dict(round=round_idx, tenant=tenant, decode_tps=r["decode_tokens_per_s"]))
        print(f"round {round_idx:2d} [{tenant}]: {r['decode_tokens_per_s']:.2f} tok/s")

    cross = sum(1 for evicted_key, evicting_key in shared.eviction_log
                if evicted_key.split(":")[0] != evicting_key.split(":")[0])
    print(f"\n[shared cache] hits={shared.hits} misses={shared.misses} "
          f"hit_rate={shared.hit_rate:.3f} evictions={shared.evictions} cross_tenant={cross} "
          f"prefetches_triggered={prefetches_triggered}/{len(SCHEDULE)}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / f"{ts}_{args.tag}_{args.scheme}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "rounds.json", "w") as f:
        json.dump(round_results, f, indent=2)
    with open(out_dir / "cache_summary.json", "w") as f:
        json.dump(dict(hits=shared.hits, misses=shared.misses, hit_rate=shared.hit_rate,
                        evictions=shared.evictions, cross_tenant_evictions=cross,
                        prefetches_triggered=prefetches_triggered), f, indent=2)
    print(f"\nResults saved to {out_dir}")

    manager_a.remove_hooks()
    manager_b.remove_hooks()
    server.stop()


if __name__ == "__main__":
    main()
