"""Ramer Local V0: find the actual max-throughput GPU/CPU layer split for a
local Ollama model on this machine, instead of trusting Ollama's own
conservative auto-placement or guessing a num_gpu by hand.

Directly productizes Experiment B's finding (see PRODUCT_HYPOTHESES.md):
llama.cpp's own offload is smooth within physical VRAM, Ollama's auto
default leaves real throughput on the table, and pushing num_gpu past the
true VRAM ceiling falls off a sharp, non-obvious cliff. This probes a
coarse grid of num_gpu values with short real generations, picks the
best-observed split, refines once around it, and caches the result per
model+GPU so repeat runs are instant.

Usage:
    python scripts/ramer_local.py probe <model> [--num-predict 40] [--force]
    python scripts/ramer_local.py run <model> <prompt...>
    python scripts/ramer_local.py show-cache
"""

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

OLLAMA_HOST = "http://localhost:11434"
CACHE_PATH = Path(__file__).resolve().parents[1] / "ramer_local_cache.json"
REGRESSION_TOL = 0.85  # a candidate more than this far below the running best is treated as a cliff, not noise


def _post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"{OLLAMA_HOST}{path}", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def _get(path: str) -> dict:
    with urllib.request.urlopen(f"{OLLAMA_HOST}{path}", timeout=30) as resp:
        return json.loads(resp.read())


def gpu_vram_mib() -> tuple[int, int]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[0]
    total, free = (int(x.strip()) for x in out.split(","))
    return total, free


def gpu_name() -> str:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[0]
    return out.strip()


def total_layers(model: str) -> int:
    info = _post("/api/show", {"model": model}).get("model_info", {})
    for key, value in info.items():
        if key.endswith(".block_count"):
            return int(value)
    raise RuntimeError(f"could not find block_count in model_info for {model}: {list(info)}")


def model_digest(model: str) -> str:
    for entry in _get("/api/tags").get("models", []):
        if entry.get("name") == model:
            return entry.get("digest", "")[:12]
    return ""


def unload(model: str):
    _post("/api/generate", {"model": model, "keep_alive": 0})


def measure(model: str, num_gpu: int | None, num_predict: int, prompt: str) -> dict:
    """One clean measurement: unload, then generate with an explicit (or
    absent, for Ollama's own auto choice) num_gpu."""
    unload(model)
    options = {"num_predict": num_predict}
    if num_gpu is not None:
        options["num_gpu"] = num_gpu
    resp = _post("/api/generate", {"model": model, "prompt": prompt, "stream": False, "options": options})
    ps = _get("/api/ps")
    size_vram = 0
    for entry in ps.get("models", []):
        if entry.get("name") == model:
            size_vram = entry.get("size_vram", 0)
    eval_count = resp.get("eval_count") or 0
    eval_duration_ns = resp.get("eval_duration") or 1
    return dict(
        num_gpu=num_gpu,
        decode_tok_s=round(eval_count / (eval_duration_ns / 1e9), 3) if eval_count else 0.0,
        size_vram_gb=round(size_vram / 1e9, 3),
    )


def probe(model: str, num_predict: int, prompt: str, log=print) -> dict:
    n_layers = total_layers(model)
    unload(model)  # in case it's still resident from a previous invocation -- keep the free-VRAM baseline honest
    _, free_at_probe_mib = gpu_vram_mib()
    log(f"[ramer-local] {model}: {n_layers} transformer blocks, probing on {gpu_name()} "
        f"({free_at_probe_mib} MiB free)...")

    auto = measure(model, None, num_predict, prompt)
    log(f"  auto (Ollama default): {auto['decode_tok_s']} tok/s, {auto['size_vram_gb']} GB VRAM")

    # Ollama's num_gpu counts transformer blocks PLUS the output/embedding
    # tensor -- block_count alone (n_layers) is one short of "fully
    # offloaded". Sweep up to n_layers+1 so the fits-in-VRAM case is
    # actually represented in the grid, not silently one layer short of
    # what auto achieves. Discovered by comparing this tool's own first
    # real run against Ollama's auto default -- auto was ~20% faster than
    # explicit num_gpu=n_layers until this fix.
    full = n_layers + 1
    steps = 6
    grid = sorted({round(i * full / steps) for i in range(steps + 1)} | {full})
    results = [dict(auto, num_gpu="auto")]
    for n in grid:
        r = measure(model, n, num_predict, prompt)
        results.append(r)
        log(f"  num_gpu={n:>3}: {r['decode_tok_s']:>7} tok/s, {r['size_vram_gb']:>6} GB VRAM")

    best = max(results, key=lambda r: r["decode_tok_s"])
    best_idx = grid.index(best["num_gpu"]) if best["num_gpu"] in grid else len(grid) - 1

    # one refinement pass: check the midpoint between the best grid point and
    # its next-higher neighbor -- catches a finer-grained optimum without a
    # full re-sweep, and confirms whether that neighbor was a real cliff.
    if best_idx + 1 < len(grid):
        mid = (grid[best_idx] + grid[best_idx + 1]) // 2
        if mid not in grid:
            r = measure(model, mid, num_predict, prompt)
            results.append(r)
            log(f"  refine num_gpu={mid:>3}: {r['decode_tok_s']:>7} tok/s, {r['size_vram_gb']:>6} GB VRAM")
            if r["decode_tok_s"] > best["decode_tok_s"]:
                best = r

    if auto["decode_tok_s"] > 0:
        gain_pct = round(100 * (best["decode_tok_s"] / auto["decode_tok_s"] - 1), 1)
    else:
        gain_pct = None

    return dict(
        model=model, n_layers=n_layers, num_predict=num_predict,
        gpu=gpu_name(), auto=auto, grid_results=results, recommended=best, gain_over_auto_pct=gain_pct,
        free_vram_mib_at_probe=free_at_probe_mib,
    )


def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def save_cache(cache: dict):
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


def cache_key(model: str, digest: str, num_predict: int) -> str:
    total_mib, _ = gpu_vram_mib()
    return f"{model}|{digest}|{gpu_name()}|{total_mib}mib|np{num_predict}"


PROBE_PROMPT = "Explain the concept of gradient descent in machine learning, covering the intuition, the update rule, and common pitfalls."


def cmd_probe(args):
    digest = model_digest(args.model)
    key = cache_key(args.model, digest, args.num_predict)
    cache = load_cache()
    if key in cache and not args.force:
        print(f"[ramer-local] cached result for {key}:")
        print(json.dumps(cache[key], indent=2))
        print("\n(use --force to re-probe)")
        return
    result = probe(args.model, args.num_predict, PROBE_PROMPT)
    cache[key] = result
    save_cache(cache)
    print(f"\n[ramer-local] recommended num_gpu={result['recommended']['num_gpu']} "
          f"({result['recommended']['decode_tok_s']} tok/s, {result['recommended']['size_vram_gb']} GB VRAM)"
          + (f" -- {result['gain_over_auto_pct']:+.1f}% vs Ollama's own auto placement" if result['gain_over_auto_pct'] is not None else ""))
    print(f"[ramer-local] cached under key: {key}")


def cmd_run(args):
    digest = model_digest(args.model)
    key = cache_key(args.model, digest, args.num_predict)
    cache = load_cache()
    freshly_probed = key not in cache or args.force
    if freshly_probed:
        print(f"[ramer-local] {'re-' if key in cache else ''}probing before run...")
        result = probe(args.model, args.num_predict, PROBE_PROMPT)
        cache[key] = result
        save_cache(cache)
    else:
        result = cache[key]
    num_gpu = result["recommended"]["num_gpu"]
    probed_free = result.get("free_vram_mib_at_probe")
    # Only meaningful against a REUSED cached result -- right after a fresh
    # probe the model is still resident from the probe's own last
    # measurement, which would make "free VRAM now" read artificially low
    # and falsely warn every single time.
    if not freshly_probed and probed_free is not None and num_gpu not in ("auto", None):
        unload(args.model)
        _, free_now = gpu_vram_mib()
        # the split was picked close to this machine's VRAM ceiling at probe
        # time -- if something else is now using noticeably more VRAM than
        # it was then, that same split can land in the cliff Experiment B
        # found, not just run a bit slower. Warn rather than block: still
        # cheaper than silently eating a 3x regression.
        if free_now < probed_free - 500:
            print(f"[ramer-local] warning: {probed_free - free_now} MiB less VRAM free now than at probe time "
                  f"({free_now} vs {probed_free} MiB) -- this split may now be past the cliff. "
                  f"Consider re-probing with --force.\n")
    prompt = " ".join(args.prompt) or PROBE_PROMPT
    print(f"[ramer-local] running {args.model} with num_gpu={num_gpu} (from cache)...\n")
    unload(args.model)
    options = {} if num_gpu == "auto" else {"num_gpu": num_gpu}
    resp = _post("/api/generate", {
        "model": args.model, "prompt": prompt, "stream": False,
        "options": options,
    })
    print(resp.get("response", ""))
    eval_count = resp.get("eval_count") or 0
    eval_duration_ns = resp.get("eval_duration") or 1
    print(f"\n[ramer-local] {round(eval_count / (eval_duration_ns / 1e9), 2)} tok/s")


def cmd_show_cache(args):
    cache = load_cache()
    if not cache:
        print("[ramer-local] cache is empty")
        return
    for key, result in cache.items():
        r = result["recommended"]
        print(f"{key}\n  num_gpu={r['num_gpu']}  {r['decode_tok_s']} tok/s  {r['size_vram_gb']} GB VRAM"
              + (f"  ({result['gain_over_auto_pct']:+.1f}% vs auto)" if result.get('gain_over_auto_pct') is not None else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_probe = sub.add_parser("probe", help="sweep num_gpu and cache the best split for a model")
    p_probe.add_argument("model")
    p_probe.add_argument("--num-predict", type=int, default=40)
    p_probe.add_argument("--force", action="store_true", help="re-probe even if cached")
    p_probe.set_defaults(func=cmd_probe)

    p_run = sub.add_parser("run", help="run a model at its cached (or newly probed) optimal split")
    p_run.add_argument("model")
    p_run.add_argument("prompt", nargs="*")
    p_run.add_argument("--num-predict", type=int, default=40, help="must match the value used for probing to hit the cache")
    p_run.add_argument("--force", action="store_true", help="re-probe even if cached")
    p_run.set_defaults(func=cmd_run)

    p_cache = sub.add_parser("show-cache", help="list all cached placements")
    p_cache.set_defaults(func=cmd_show_cache)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
