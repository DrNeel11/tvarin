"""Phase 3 entrypoint: layer-level tiered memory, synchronous fetch-on-demand.

Loads a model that normally fits entirely in VRAM (Qwen2.5-1.5B by default),
then artificially forces its *last* N layers to live in RAM or on the
simulated remote server instead, evicting them back off the GPU after every
single use. This reproduces, at individually-instrumented layer
granularity, the same "doesn't fit -> falls off a cliff" pattern Phase 1
observed from accelerate's opaque CPU offload (7B: 27 -> 1.1 tok/s) -- and
gives Phase 4 something concrete to improve on by prefetching ahead instead
of fetching synchronously.

Usage:
    python scripts/run_tiered.py --overflow-layers 8 --overflow-tier remote \
        --latency-ms 10 --bandwidth-mb-per-s 100
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
from memtier.tiered.model_introspection import find_decoder_layers
from memtier.tiered.safetensors_reader import SafetensorsMultiReader
from memtier.tiered.tier_manager import LayerTierManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-label", default="qwen2.5-1.5b-fp16")
    parser.add_argument("--overflow-layers", type=int, default=8)
    parser.add_argument("--overflow-tier", choices=["ram", "remote"], default="remote")
    parser.add_argument("--placement", choices=["manual", "auto"], default="manual",
                         help="manual: first N layers gpu-resident, rest overflow_tier (small models). "
                              "auto: use accelerate's own device_map placement for gpu/cpu split (large "
                              "models that need max_memory-capped loading), then --remote-layers of the "
                              "CPU-resident layers are redirected to the remote tier, rest stay 'ram'.")
    parser.add_argument("--remote-layers", type=int, default=None,
                         help="auto placement only: how many of the CPU-resident layers to redirect to remote")
    parser.add_argument("--latency-ms", type=float, default=10.0)
    parser.add_argument("--bandwidth-mb-per-s", type=float, default=100.0)
    parser.add_argument("--n-reps", type=int, default=3)
    parser.add_argument("--n-warmup", type=int, default=1)
    parser.add_argument("--prefetch", action="store_true", help="Phase 4: prefetch layer N+1 while layer N computes")
    parser.add_argument("--cache-after-fetch", action="store_true",
                         help="Phase 5: keep fetched layers resident in RAM instead of evicting them")
    parser.add_argument("--ram-cache-budget-mb", type=float, default=None,
                         help="cap on persistent RAM cache size; None = unlimited")
    parser.add_argument("--batch-size", type=int, default=1,
                         help="replicate the prompt across the batch dim to test whether prefetch "
                              "overlap improves as compute-per-layer grows relative to transfer-per-layer")
    parser.add_argument("--tag", default="phase3_tiered")
    args = parser.parse_args()

    print(f"CUDA available: {torch.cuda.is_available()}")
    cfg = get_by_label(args.model_label)

    model, tokenizer, load_time_s, device_map_summary = load_model(cfg)
    print(f"loaded {args.model_label} in {load_time_s:.1f}s ({device_map_summary})")

    decoder_layers, layer_key_prefix = find_decoder_layers(model)
    n_layers = len(decoder_layers)
    print(f"decoder layers found at '{layer_key_prefix}' ({type(model).__name__})")

    state_loader = None
    if args.placement == "manual":
        n_gpu = n_layers - args.overflow_layers
        if n_gpu < 1:
            raise ValueError(f"overflow_layers={args.overflow_layers} leaves no resident layers (model has {n_layers})")
        tier_map = {i: "gpu" for i in range(n_gpu)}
        tier_map.update({i: args.overflow_tier for i in range(n_gpu, n_layers)})
        n_remote = args.overflow_layers if args.overflow_tier == "remote" else 0
        print(f"model has {n_layers} layers: {n_gpu} resident on GPU, "
              f"{args.overflow_layers} on {args.overflow_tier}")
    else:
        # Trust accelerate's own device_map (from a max_memory-capped load)
        # to decide gpu vs cpu placement -- this is the same well-tested
        # path Phase 1 used, so it correctly handles buffers/rotary
        # embeddings etc. that a hand-rolled loader would risk getting
        # wrong. We only add a second decision on top: which of the
        # CPU-resident layers get redirected to the remote tier.
        device_map = model.hf_device_map
        cpu_layers = sorted(
            i for i in range(n_layers)
            if str(device_map.get(f"{layer_key_prefix}.{i}", "cpu")) == "cpu"
        )
        gpu_layers = [i for i in range(n_layers) if i not in cpu_layers]
        if args.remote_layers is None:
            raise ValueError("--placement auto requires --remote-layers")
        if args.remote_layers > len(cpu_layers):
            raise ValueError(
                f"--remote-layers {args.remote_layers} exceeds the {len(cpu_layers)} "
                f"CPU-resident layers accelerate actually placed"
            )
        remote_set = set(cpu_layers[-args.remote_layers:]) if args.remote_layers > 0 else set()
        tier_map = {}
        for i in range(n_layers):
            if i in remote_set:
                tier_map[i] = "remote"
            elif i in cpu_layers:
                tier_map[i] = "ram"
            else:
                tier_map[i] = "gpu"
        n_gpu = len(gpu_layers)
        n_remote = len(remote_set)
        n_ram = len(cpu_layers) - n_remote
        print(f"model has {n_layers} layers: accelerate placed {n_gpu} on GPU, "
              f"{len(cpu_layers)} on CPU -- of those, {n_remote} redirected to remote, "
              f"{n_ram} stay as always-resident RAM")

        # accelerate's own dispatch represents CPU-offloaded layers as
        # meta-tensor placeholders backed by its own hooks -- extracting
        # real data from the live module fails ("Cannot copy out of meta
        # tensor"), and leaving its hooks attached would race with ours.
        # Detach its hooks now (cheap); the actual checkpoint reads happen
        # lazily below, one layer at a time, so no more than one layer's
        # weights are ever double-buffered in memory at once -- reading
        # all ~41 layers eagerly here previously segfaulted the process
        # (each layer's data existed twice: once in a staging dict, once
        # copied into the manager's own cache, at ~22.6GB total).
        import accelerate.hooks
        for i in cpu_layers:
            accelerate.hooks.remove_hook_from_module(decoder_layers[i], recurse=True)
        reader = SafetensorsMultiReader(cfg.repo_id)

        def state_loader(idx):
            return reader.read_layer_state(idx, prefix=layer_key_prefix)

    remote_client = None
    prefetch_remote_client = None
    server = None
    if n_remote > 0:
        server = RemoteMemoryServer(
            host="127.0.0.1", port=5970, latency_ms=args.latency_ms,
            bandwidth_mb_per_s=args.bandwidth_mb_per_s,
        )
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        time.sleep(0.3)
        remote_client = RemoteMemoryClient(
            host="127.0.0.1", port=5970, upload_bandwidth_mb_per_s=args.bandwidth_mb_per_s,
        )
        remote_client.connect()
        if args.prefetch:
            # Separate connection so the background prefetch thread's GET
            # doesn't queue behind the foreground fetch on the same socket.
            prefetch_remote_client = RemoteMemoryClient(
                host="127.0.0.1", port=5970, upload_bandwidth_mb_per_s=args.bandwidth_mb_per_s,
            )
            prefetch_remote_client.connect()

    torch.cuda.synchronize()
    vram_before_partition = torch.cuda.memory_allocated() / 1e9

    ram_cache_budget_bytes = (
        args.ram_cache_budget_mb * 1_000_000 if args.ram_cache_budget_mb is not None else None
    )
    gpu_device = torch.device("cuda:0")
    manager = LayerTierManager(
        decoder_layers, tier_map, device=gpu_device,
        remote_client=remote_client, run_id=args.tag,
        enable_prefetch=args.prefetch, prefetch_remote_client=prefetch_remote_client,
        cache_after_fetch=args.cache_after_fetch, ram_cache_budget_bytes=ram_cache_budget_bytes,
        state_loader=state_loader,
    )
    manager.install_hooks()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    vram_after_partition = torch.cuda.memory_allocated() / 1e9
    if args.placement == "manual":
        print(f"VRAM: {vram_before_partition:.2f}GB (all-resident) -> "
              f"{vram_after_partition:.2f}GB (after offloading {args.overflow_layers} layers)")
    else:
        print(f"VRAM: {vram_after_partition:.2f}GB (accelerate's load already placed only "
              f"{n_gpu} layers on GPU; redirecting {n_remote} CPU layers to remote changes RAM, not VRAM)")

    prompt_results = []
    for prompt_name, prompt_text in BENCHMARK_PROMPTS.items():
        for _ in range(args.n_warmup):
            run_one_prompt(model, tokenizer, prompt_name, prompt_text, batch_size=args.batch_size)

        reps = [
            run_one_prompt(model, tokenizer, prompt_name, prompt_text, batch_size=args.batch_size)
            for _ in range(args.n_reps)
        ]
        decode_vals = sorted(r["decode_tokens_per_s"] for r in reps if r["decode_tokens_per_s"])
        median_decode = decode_vals[len(decode_vals) // 2] if decode_vals else None
        ttft_vals = sorted(r["ttft_s"] for r in reps if r["ttft_s"] is not None)
        median_ttft = ttft_vals[len(ttft_vals) // 2] if ttft_vals else None

        print(f"[{prompt_name}] median decode={median_decode:.2f}tok/s ttft={median_ttft:.3f}s")
        prompt_results.append(dict(
            prompt_name=prompt_name, median_decode_tokens_per_s=median_decode,
            median_ttft_s=median_ttft, n_reps=args.n_reps,
        ))

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / f"{ts}_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args) | dict(
            n_layers=n_layers, n_gpu_resident=n_gpu, n_remote_actual=n_remote,
            vram_before_gb=vram_before_partition, vram_after_gb=vram_after_partition,
        ), f, indent=2)
    with open(out_dir / "prompt_results.json", "w") as f:
        json.dump(prompt_results, f, indent=2)

    layer_op_rows = [vars(op) for op in manager.ops]
    if layer_op_rows:
        with open(out_dir / "layer_ops.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(layer_op_rows[0].keys()))
            writer.writeheader()
            writer.writerows(layer_op_rows)
        fetch_ops = [r for r in layer_op_rows if r["op"] == "fetch"]
        avg_fetch_wall = sum(r["wall_s"] for r in fetch_ops) / len(fetch_ops)
        avg_wait = sum(r["fetch_wait_s"] or 0 for r in fetch_ops) / len(fetch_ops)
        print(f"\n{len(fetch_ops)} layer fetches, avg wall time per fetch: {avg_fetch_wall*1000:.2f}ms, "
              f"avg blocked-wait per fetch: {avg_wait*1000:.2f}ms")
        if args.prefetch:
            prefetched_ops = [r for r in fetch_ops if r["prefetched"]]
            fully_hidden = [r for r in prefetched_ops if (r["fetch_wait_s"] or 0) < 0.001]
            print(f"prefetched: {len(prefetched_ops)}/{len(fetch_ops)} fetches "
                  f"({len(fully_hidden)} fully hidden, wait<1ms)")
        if args.cache_after_fetch:
            cache_hits = [r for r in fetch_ops if r["ram_cache_hit"]]
            print(f"RAM cache hits: {len(cache_hits)}/{len(fetch_ops)} fetches "
                  f"({len(cache_hits)/len(fetch_ops)*100:.1f}%)")

    print(f"\nResults saved to {out_dir}")

    if server is not None:
        server.stop()
    manager.remove_hooks()


if __name__ == "__main__":
    main()
