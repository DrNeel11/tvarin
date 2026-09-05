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

OLLAMA_HOST = "http://127.0.0.1:11434"  # NOT "localhost" -- urllib resolving it stalls ~2s/request on this machine (tries IPv6 first, falls back), found by timing every request in a single `run` call summing to ~22s of pure overhead unrelated to the model or GPU at all
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


def unload_others(target_model: str):
    """Ollama's default keep_alive (~5 min) keeps a previously-used model
    resident in VRAM after you've moved on -- confirmed directly: switching
    models and back four times in a row, every switch back to a cached
    placement measured well below its probe-time throughput, because the
    prior model was still warm and competing for VRAM during the load
    transition. unload() only ever targeted the model about to run; nothing
    evicted whatever else was left over from the previous call."""
    for entry in _get("/api/ps").get("models", []):
        name = entry.get("name")
        if name and name != target_model:
            unload(name)


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


def safe_measure(model: str, num_gpu, num_predict: int, prompt: str) -> dict:
    """measure(), but a candidate that crashes Ollama server-side (observed
    directly: qwen2.5:32b-instruct at a high num_gpu returned HTTP 500 --
    server survived, that one request didn't) is data, not a reason to kill
    the whole sweep. Recorded as a failed candidate (decode_tok_s=0) so it's
    never picked as best, same treatment as a candidate that merely falls
    off the VRAM cliff."""
    try:
        return dict(measure(model, num_gpu, num_predict, prompt), failed=False)
    except Exception as e:
        return dict(num_gpu=num_gpu, decode_tok_s=0.0, size_vram_gb=None, failed=True, error=str(e))


def probe(model: str, num_predict: int, prompt: str, log=print) -> dict:
    n_layers = total_layers(model)
    unload(model)
    unload_others(model)  # a warm leftover model from a previous invocation would make this probe's own VRAM baseline wrong
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
    results = [dict(auto, num_gpu="auto", failed=False)]
    for n in grid:
        r = safe_measure(model, n, num_predict, prompt)
        results.append(r)
        if r["failed"]:
            log(f"  num_gpu={n:>3}: FAILED ({r['error']}) -- stopping sweep, higher splits won't recover")
            break
        log(f"  num_gpu={n:>3}: {r['decode_tok_s']:>7} tok/s, {r['size_vram_gb']:>6} GB VRAM")

    ok_results = [r for r in results if not r["failed"]]
    best = max(ok_results, key=lambda r: r["decode_tok_s"])
    best_idx = grid.index(best["num_gpu"]) if best["num_gpu"] in grid else len(grid) - 1

    # one refinement pass: check the midpoint between the best grid point and
    # its next-higher neighbor -- catches a finer-grained optimum without a
    # full re-sweep, and confirms whether that neighbor was a real cliff.
    # Skipped if the sweep above already stopped early on a failure --
    # nothing above the best point is trustworthy to probe further.
    if best_idx + 1 < len(grid) and len(ok_results) == len(results):
        mid = (grid[best_idx] + grid[best_idx + 1]) // 2
        if mid not in grid:
            r = safe_measure(model, mid, num_predict, prompt)
            results.append(r)
            if r["failed"]:
                log(f"  refine num_gpu={mid:>3}: FAILED ({r['error']})")
            else:
                log(f"  refine num_gpu={mid:>3}: {r['decode_tok_s']:>7} tok/s, {r['size_vram_gb']:>6} GB VRAM")
            if not r["failed"] and r["decode_tok_s"] > best["decode_tok_s"]:
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


QUICK_CHECK_PREDICT = 20
QUICK_CHECK_REGRESSION_TOL = 0.7  # below this fraction of the cached tok/s counts as "the split is now stale"
USAGE_LOG_PATH = Path(__file__).resolve().parents[1] / "ramer_local_usage.jsonl"


def log_usage(**fields):
    fields["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(USAGE_LOG_PATH, "a") as f:
        f.write(json.dumps(fields) + "\n")


def cmd_run(args):
    unload_others(args.model)  # a model left warm from a previous run competes with this one for VRAM -- see unload_others()
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
    cached_tps = result["recommended"]["decode_tok_s"]
    stale = False

    # Only meaningful against a REUSED cached result -- right after a fresh
    # probe the model is still resident from the probe's own last
    # measurement, so a check here would just re-confirm the number we
    # already trust. A quick direct throughput check is ground truth
    # (whatever changed -- VRAM pressure, something else on the GPU --
    # doesn't need to be identified, only detected) rather than inferring
    # staleness from free-VRAM alone.
    if not freshly_probed and num_gpu not in ("auto", None):
        check = measure(args.model, num_gpu, QUICK_CHECK_PREDICT, PROBE_PROMPT)
        if cached_tps > 0 and check["decode_tok_s"] < cached_tps * QUICK_CHECK_REGRESSION_TOL:
            print(f"[ramer-local] cached split (num_gpu={num_gpu}) now measures {check['decode_tok_s']} tok/s, "
                  f"down from {cached_tps} tok/s at probe time -- likely past the cliff now. "
                  f"Falling back to Ollama's auto placement for this run (re-probe with --force when convenient).\n")
            num_gpu = "auto"
            stale = True

    prompt = " ".join(args.prompt) or PROBE_PROMPT
    print(f"[ramer-local] running {args.model} with num_gpu={num_gpu}...\n")
    unload(args.model)
    options = {} if num_gpu == "auto" else {"num_gpu": num_gpu}
    resp = _post("/api/generate", {
        "model": args.model, "prompt": prompt, "stream": False,
        "options": options,
    })
    print(resp.get("response", ""))
    eval_count = resp.get("eval_count") or 0
    eval_duration_ns = resp.get("eval_duration") or 1
    ramer_tps = round(eval_count / (eval_duration_ns / 1e9), 2)
    print(f"\n[ramer-local] {ramer_tps} tok/s")

    log_usage(
        model=args.model, num_gpu=num_gpu, ramer_tok_s=ramer_tps,
        auto_tok_s=result["auto"]["decode_tok_s"], stale=stale, freshly_probed=freshly_probed,
        eval_count=eval_count,
    )


def cmd_show_cache(args):
    cache = load_cache()
    if not cache:
        print("[ramer-local] cache is empty")
        return
    for key, result in cache.items():
        r = result["recommended"]
        print(f"{key}\n  num_gpu={r['num_gpu']}  {r['decode_tok_s']} tok/s  {r['size_vram_gb']} GB VRAM"
              + (f"  ({result['gain_over_auto_pct']:+.1f}% vs auto)" if result.get('gain_over_auto_pct') is not None else ""))


def cmd_show_log(args):
    if not USAGE_LOG_PATH.exists():
        print("[ramer-local] usage log is empty -- run `ramer_local.py run <model> ...` a few times first")
        return
    rows = [json.loads(line) for line in USAGE_LOG_PATH.read_text().splitlines() if line.strip()]
    print(f"{'ts':<20} {'model':<28} {'num_gpu':>8} {'ramer tok/s':>12} {'auto tok/s':>11} {'gain':>7}  stale")
    for r in rows:
        gain = f"{100 * (r['ramer_tok_s'] / r['auto_tok_s'] - 1):+.1f}%" if r.get("auto_tok_s") else "n/a"
        print(f"{r['ts']:<20} {r['model']:<28} {str(r['num_gpu']):>8} {r['ramer_tok_s']:>12} "
              f"{r.get('auto_tok_s', 'n/a'):>11} {gain:>7}  {'yes' if r.get('stale') else ''}")


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

    p_log = sub.add_parser("show-log", help="show the history of every `run` invocation (auto vs ramer tok/s)")
    p_log.set_defaults(func=cmd_show_log)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
