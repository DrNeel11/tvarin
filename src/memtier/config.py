"""Global config: cache locations and fixed benchmark prompts.

HF cache is pinned to D:\\ because C:\\ only has ~26GB free on this machine
and the model ladder downloads ~50GB+ of weights.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HF_CACHE_DIR = PROJECT_ROOT / ".hf_cache"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

# Fixed prompt set: short/medium/long, so prefill-vs-decode behavior can be
# separated from prompt length effects. Kept as plain instruction-style text
# since all Phase 1 models are *-Instruct checkpoints.
BENCHMARK_PROMPTS: dict[str, str] = {
    "short": "What is the capital of France?",
    "medium": (
        "Summarize the following in three sentences: The transformer "
        "architecture, introduced in 2017, replaced recurrent and "
        "convolutional layers with self-attention, allowing models to "
        "process sequences in parallel rather than step by step. This "
        "parallelism, combined with the ability to model long-range "
        "dependencies directly, made transformers substantially more "
        "efficient to train at scale, and they have since become the "
        "dominant architecture for language, vision, and multimodal models."
    ),
    "long": (
        "Write a detailed step-by-step explanation of how a computer's "
        "memory hierarchy (registers, cache, RAM, disk) trades off speed "
        "against capacity, and explain why this same tradeoff motivates "
        "using cloud/remote memory as an additional, slower tier for "
        "workloads whose data does not fit in local RAM or VRAM. Cover "
        "at least four distinct tiers and discuss the latency and "
        "bandwidth characteristics of each."
    ),
}

MAX_NEW_TOKENS = 128
N_REPETITIONS = 5
N_WARMUP = 1
