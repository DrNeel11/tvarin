"""Persist and summarize Phase 1 benchmark results."""

from __future__ import annotations

import csv
import json
import platform
import time
from pathlib import Path

import psutil
import torch

from memtier.benchmark.harness import RunResult
from memtier.config import RESULTS_DIR


def system_info() -> dict:
    info = {
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_total_bytes": psutil.virtual_memory().total,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_vram_total_bytes"] = torch.cuda.get_device_properties(0).total_memory
        info["gpu_compute_capability"] = torch.cuda.get_device_capability(0)
    return info


def save_results(results: list[RunResult], run_tag: str | None = None) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_tag = run_tag or "phase1_baseline"
    out_dir = RESULTS_DIR / f"{ts}_{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "system_info.json", "w") as f:
        json.dump(system_info(), f, indent=2, default=str)

    rows = [r.to_dict() for r in results]
    with open(out_dir / "results.json", "w") as f:
        json.dump(rows, f, indent=2)

    if rows:
        fieldnames = list(rows[0].keys())
        with open(out_dir / "results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return out_dir


def print_summary(results: list[RunResult]):
    header = (
        f"{'label':<20}{'prompt':<8}{'status':<8}{'ttft_s':>8}"
        f"{'decode_tok/s':>14}{'prefill_tok/s':>14}{'vram_GB':>10}{'ram_GB':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        vram_gb = r.peak_vram_bytes / 1e9 if r.peak_vram_bytes else 0
        ram_gb = r.peak_ram_bytes / 1e9 if r.peak_ram_bytes else 0
        ttft = f"{r.ttft_s:.3f}" if r.ttft_s is not None else "-"
        decode = f"{r.decode_tokens_per_s:.1f}" if r.decode_tokens_per_s else "-"
        prefill = f"{r.prompt_tokens_per_s:.1f}" if r.prompt_tokens_per_s else "-"
        print(
            f"{r.label:<20}{r.prompt_name:<8}{r.status:<8}{ttft:>8}"
            f"{decode:>14}{prefill:>14}{vram_gb:>10.2f}{ram_gb:>10.2f}"
        )
