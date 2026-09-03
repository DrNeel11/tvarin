"""Phase 1 entrypoint: run the full model ladder and report results.

Usage:
    python scripts/run_baseline.py                 # full ladder
    python scripts/run_baseline.py --labels qwen2.5-0.5b-fp16 qwen2.5-1.5b-fp16
"""

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import memtier.config  # noqa: F401  (sets HF_HOME before transformers/huggingface_hub import)
import torch

from memtier.benchmark.harness import benchmark_config
from memtier.benchmark.report import print_summary, save_results
from memtier.config import BENCHMARK_PROMPTS
from memtier.models.registry import MODEL_LADDER, get_by_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--tag", default="phase1_baseline")
    parser.add_argument("--n-reps", type=int, default=None, help="override N_REPETITIONS")
    parser.add_argument("--n-warmup", type=int, default=None, help="override N_WARMUP")
    args = parser.parse_args()

    configs = (
        [get_by_label(l) for l in args.labels] if args.labels else MODEL_LADDER
    )

    from memtier.config import N_REPETITIONS, N_WARMUP
    n_reps = args.n_reps if args.n_reps is not None else N_REPETITIONS
    n_warmup = args.n_warmup if args.n_warmup is not None else N_WARMUP
    if args.n_reps is not None or args.n_warmup is not None:
        print(
            f"WARNING: reduced-fidelity run (n_reps={n_reps}, n_warmup={n_warmup}) "
            f"-- results are not a robust median, treat as anecdotal"
        )

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Compute capability: {torch.cuda.get_device_capability(0)}")

    all_results = []
    for cfg in configs:
        results = benchmark_config(cfg, BENCHMARK_PROMPTS, n_reps=n_reps, n_warmup=n_warmup)
        all_results.extend(results)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_dir = save_results(all_results, run_tag=args.tag)
    print(f"\nResults saved to {out_dir}\n")
    print_summary(all_results)


if __name__ == "__main__":
    main()
