"""Model ladder for Phase 1 baseline benchmarking.

Qwen2.5-Instruct family chosen because it spans a wide size range with a
single consistent architecture, so "size" is the only variable changing
across runs. All entries are ungated on Hugging Face.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    label: str          # unique id for this run config, e.g. "qwen2.5-7b-fp16"
    repo_id: str        # HF repo id
    param_count_b: float  # approx parameter count in billions
    quant: str           # "fp16" | "bf16" | "nf4" | "int8"
    extra_load_kwargs: dict = field(default_factory=dict)


# Explicit memory caps for the larger configs rather than trusting
# device_map="auto" to leave enough headroom on a 32GB machine: an
# uncontrolled system-level OOM (as opposed to a caught CUDA OOM) can make
# Windows page heavily and hang, not just fail cleanly.
_BIG_MAX_MEMORY = {0: "7GiB", "cpu": "22GiB"}

MODEL_LADDER: list[ModelConfig] = [
    ModelConfig("qwen2.5-0.5b-fp16", "Qwen/Qwen2.5-0.5B-Instruct", 0.5, "bf16"),
    ModelConfig("qwen2.5-1.5b-fp16", "Qwen/Qwen2.5-1.5B-Instruct", 1.5, "bf16"),
    ModelConfig("qwen2.5-3b-fp16", "Qwen/Qwen2.5-3B-Instruct", 3.0, "bf16"),
    ModelConfig(
        "qwen2.5-7b-fp16", "Qwen/Qwen2.5-7B-Instruct", 7.0, "bf16",
        extra_load_kwargs={"max_memory": _BIG_MAX_MEMORY},
    ),
    ModelConfig(
        "qwen2.5-7b-nf4", "Qwen/Qwen2.5-7B-Instruct", 7.0, "nf4",
        extra_load_kwargs={"max_memory": _BIG_MAX_MEMORY},
    ),
    ModelConfig(
        "qwen2.5-14b-fp16", "Qwen/Qwen2.5-14B-Instruct", 14.0, "bf16",
        extra_load_kwargs={"max_memory": _BIG_MAX_MEMORY},
    ),
    ModelConfig(
        "qwen2.5-14b-nf4", "Qwen/Qwen2.5-14B-Instruct", 14.0, "nf4",
        extra_load_kwargs={"max_memory": _BIG_MAX_MEMORY},
    ),
    # nf4 alone doesn't fit this 8GB card for 14B (~7.5GB weights vs ~7GB
    # truly free VRAM); bitsandbytes' fp32 CPU-offload fallback lets the
    # overflow spill to CPU instead of failing to load outright.
    ModelConfig(
        "qwen2.5-14b-nf4-cpuoffload", "Qwen/Qwen2.5-14B-Instruct", 14.0, "nf4_cpu_offload",
        extra_load_kwargs={"max_memory": _BIG_MAX_MEMORY},
    ),
    # For the Phase 3-5 tiered-memory experiment on the real 14B model:
    # a bigger CPU cap than _BIG_MAX_MEMORY's 22GiB, specifically sized so
    # accelerate's own placement doesn't need disk offload at all (Phase 1
    # measured ~551MB/layer bf16; 48 layers total). Leaves ~6GB of system
    # RAM headroom (32GB machine) plus whatever the remote-server process
    # needs for the layers we redirect to it after loading.
    ModelConfig(
        "qwen2.5-14b-fp16-tiered", "Qwen/Qwen2.5-14B-Instruct", 14.0, "bf16",
        extra_load_kwargs={"max_memory": {0: "6.5GiB", "cpu": "26GiB"}},
    ),
    # Phase 9: GPT-NeoX architecture (gpt_neox.layers, not model.layers) --
    # deliberately a different family than the Qwen2/Llama-style models
    # above, to test find_decoder_layers() against something the original
    # hardcoded model.model.layers access would have failed on.
    ModelConfig("pythia-160m-fp16", "EleutherAI/pythia-160m", 0.16, "bf16"),
]


def get_by_label(label: str) -> ModelConfig:
    for cfg in MODEL_LADDER:
        if cfg.label == label:
            return cfg
    raise KeyError(f"no model config with label {label!r}")
