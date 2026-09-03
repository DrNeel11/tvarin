"""Phase 1 benchmark harness: load a model, run fixed prompts, measure.

Deliberately does one prefill-only forward pass (for "prompt processing
speed") separately from the generate() call used for TTFT/decode throughput,
rather than deriving prompt tok/s from TTFT. TTFT includes the first decode
step as well as prefill, so conflating the two would understate prefill
speed and overstate decode latency.
"""

from __future__ import annotations

import gc
import time
from dataclasses import asdict, dataclass

# Must precede the transformers/huggingface_hub import: HF_HOME is only
# honored if set before huggingface_hub resolves its cache-path constants at
# import time. Importing memtier.config first is what actually pins the
# cache to D:\ instead of the near-full C:\ system drive.
from memtier.config import MAX_NEW_TOKENS, N_REPETITIONS, N_WARMUP

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from memtier.benchmark.metrics import (
    GpuUtilSampler,
    cpu_percent,
    nvml_available,
    process_rss_bytes,
)
from memtier.benchmark.timing_streamer import TokenTimingStreamer
from memtier.models.registry import ModelConfig


@dataclass
class RunResult:
    label: str
    repo_id: str
    quant: str
    param_count_b: float
    prompt_name: str
    status: str  # "ok" | "oom" | "error"
    error_message: str | None = None

    device_map_summary: str | None = None
    load_time_s: float | None = None
    n_reps: int | None = None
    n_warmup: int | None = None

    prompt_tokens: int | None = None
    prompt_processing_s: float | None = None
    prompt_tokens_per_s: float | None = None

    max_new_tokens: int | None = None
    generated_tokens: int | None = None
    ttft_s: float | None = None
    decode_time_s: float | None = None
    decode_tokens_per_s: float | None = None
    total_generate_s: float | None = None
    overall_tokens_per_s: float | None = None

    peak_vram_bytes: int | None = None
    peak_ram_bytes: int | None = None
    avg_gpu_util_pct: float | None = None
    max_gpu_util_pct: float | None = None
    avg_cpu_percent: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _quant_load_kwargs(quant: str) -> dict:
    if quant in ("fp16", "bf16"):
        dtype = torch.float16 if quant == "fp16" else torch.bfloat16
        return {"dtype": dtype}
    if quant == "nf4":
        return {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        }
    if quant == "nf4_cpu_offload":
        return {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
        }
    if quant == "int8":
        return {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
    raise ValueError(f"unknown quant {quant!r}")


def load_model(cfg: ModelConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.repo_id)
    kwargs = _quant_load_kwargs(cfg.quant)
    kwargs.update(cfg.extra_load_kwargs)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        cfg.repo_id, device_map="auto", **kwargs
    )
    model.eval()
    load_time_s = time.perf_counter() - t0

    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        devices = sorted(set(str(v) for v in device_map.values()))
        device_map_summary = ",".join(devices)
    else:
        device_map_summary = str(next(model.parameters()).device)

    return model, tokenizer, load_time_s, device_map_summary


def run_one_prompt(model, tokenizer, prompt_name: str, prompt_text: str, batch_size: int = 1) -> dict:
    """batch_size > 1 replicates the same prompt across the batch dimension
    -- identical content, so no padding is needed. Used to test whether
    per-layer compute time (which scales with batch size) can grow enough
    relative to per-layer transfer time (which doesn't, since weights are
    reused across the batch) for prefetch overlap to start paying off; see
    Phase 3/4's batch=1 negative result in docs/ARCHITECTURE.md."""
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt_text}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # Base (non-instruction-tuned) models have no chat template --
        # e.g. EleutherAI/pythia-160m, used in Phase 9's cross-architecture
        # test. Fall back to the raw prompt text.
        input_text = prompt_text
    inputs = tokenizer([input_text] * batch_size, return_tensors="pt").to(model.device)
    prompt_tokens = inputs["input_ids"].shape[1]

    # 1. Prefill-only timing (prompt processing speed).
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        model(**inputs, use_cache=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prompt_processing_s = time.perf_counter() - t0
    prompt_tokens_per_s = (
        prompt_tokens / prompt_processing_s if prompt_processing_s > 0 else None
    )

    # 2. Full generation timing (TTFT + decode throughput).
    streamer = TokenTimingStreamer()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_gen_start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            streamer=streamer,
            pad_token_id=tokenizer.eos_token_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_gen_end = time.perf_counter()

    generated_tokens_per_seq = output_ids.shape[1] - prompt_tokens
    generated_tokens = generated_tokens_per_seq * batch_size  # aggregate across the batch
    ts = streamer.timestamps
    ttft_s = ts[0] - t_gen_start if ts else None
    if len(ts) >= 2:
        decode_time_s = ts[-1] - ts[0]
        decode_steps = len(ts) - 1
        decode_tokens = decode_steps * batch_size
        decode_tokens_per_s = (
            decode_tokens / decode_time_s if decode_time_s > 0 else None
        )
    else:
        decode_time_s = None
        decode_tokens_per_s = None
    total_generate_s = t_gen_end - t_gen_start
    overall_tokens_per_s = (
        generated_tokens / total_generate_s if total_generate_s > 0 else None
    )

    return dict(
        prompt_tokens=prompt_tokens,
        prompt_processing_s=prompt_processing_s,
        prompt_tokens_per_s=prompt_tokens_per_s,
        max_new_tokens=MAX_NEW_TOKENS,
        batch_size=batch_size,
        generated_tokens=generated_tokens,
        ttft_s=ttft_s,
        decode_time_s=decode_time_s,
        decode_tokens_per_s=decode_tokens_per_s,
        total_generate_s=total_generate_s,
        overall_tokens_per_s=overall_tokens_per_s,
    )


def benchmark_config(
    cfg: ModelConfig,
    prompts: dict[str, str],
    n_reps: int = N_REPETITIONS,
    n_warmup: int = N_WARMUP,
    log=print,
) -> list[RunResult]:
    results: list[RunResult] = []

    log(f"[{cfg.label}] loading...")
    try:
        model, tokenizer, load_time_s, device_map_summary = load_model(cfg)
    except torch.OutOfMemoryError as e:
        log(f"[{cfg.label}] OOM during load: {e}")
        results.append(
            RunResult(
                label=cfg.label,
                repo_id=cfg.repo_id,
                quant=cfg.quant,
                param_count_b=cfg.param_count_b,
                prompt_name="<load>",
                status="oom",
                error_message=str(e),
            )
        )
        torch.cuda.empty_cache()
        return results
    except Exception as e:
        log(f"[{cfg.label}] error during load: {e}")
        results.append(
            RunResult(
                label=cfg.label,
                repo_id=cfg.repo_id,
                quant=cfg.quant,
                param_count_b=cfg.param_count_b,
                prompt_name="<load>",
                status="error",
                error_message=str(e),
            )
        )
        return results

    log(f"[{cfg.label}] loaded in {load_time_s:.1f}s, device_map={device_map_summary}")

    for prompt_name, prompt_text in prompts.items():
        try:
            for i in range(n_warmup):
                run_one_prompt(model, tokenizer, prompt_name, prompt_text)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            rss_before = process_rss_bytes()

            rep_metrics = []
            cpu_pcts = []
            cpu_percent(interval=None)  # prime the internal baseline
            sampler = GpuUtilSampler()
            with sampler:
                for i in range(n_reps):
                    rep_metrics.append(
                        run_one_prompt(model, tokenizer, prompt_name, prompt_text)
                    )
                    cpu_pcts.append(cpu_percent(interval=None))

            def median(key):
                vals = [m[key] for m in rep_metrics if m[key] is not None]
                if not vals:
                    return None
                vals.sort()
                return vals[len(vals) // 2]

            peak_vram = (
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
            )
            peak_ram = max(rss_before, process_rss_bytes())

            results.append(
                RunResult(
                    label=cfg.label,
                    repo_id=cfg.repo_id,
                    quant=cfg.quant,
                    param_count_b=cfg.param_count_b,
                    prompt_name=prompt_name,
                    status="ok",
                    device_map_summary=device_map_summary,
                    load_time_s=load_time_s,
                    n_reps=n_reps,
                    n_warmup=n_warmup,
                    prompt_tokens=rep_metrics[0]["prompt_tokens"],
                    prompt_processing_s=median("prompt_processing_s"),
                    prompt_tokens_per_s=median("prompt_tokens_per_s"),
                    max_new_tokens=MAX_NEW_TOKENS,
                    generated_tokens=rep_metrics[0]["generated_tokens"],
                    ttft_s=median("ttft_s"),
                    decode_time_s=median("decode_time_s"),
                    decode_tokens_per_s=median("decode_tokens_per_s"),
                    total_generate_s=median("total_generate_s"),
                    overall_tokens_per_s=median("overall_tokens_per_s"),
                    peak_vram_bytes=peak_vram,
                    peak_ram_bytes=peak_ram,
                    avg_gpu_util_pct=sampler.avg_gpu_util_pct,
                    max_gpu_util_pct=sampler.max_gpu_util_pct,
                    avg_cpu_percent=sum(cpu_pcts) / len(cpu_pcts) if cpu_pcts else None,
                )
            )
            log(
                f"[{cfg.label}/{prompt_name}] "
                f"ttft={results[-1].ttft_s:.3f}s "
                f"decode={results[-1].decode_tokens_per_s or 0:.1f}tok/s "
                f"prefill={results[-1].prompt_tokens_per_s or 0:.1f}tok/s "
                f"vram_peak={peak_vram/1e9 if peak_vram else 0:.2f}GB"
            )
        except torch.OutOfMemoryError as e:
            log(f"[{cfg.label}/{prompt_name}] OOM: {e}")
            results.append(
                RunResult(
                    label=cfg.label,
                    repo_id=cfg.repo_id,
                    quant=cfg.quant,
                    param_count_b=cfg.param_count_b,
                    prompt_name=prompt_name,
                    status="oom",
                    error_message=str(e),
                )
            )
            torch.cuda.empty_cache()
        except Exception as e:
            log(f"[{cfg.label}/{prompt_name}] error: {e}")
            results.append(
                RunResult(
                    label=cfg.label,
                    repo_id=cfg.repo_id,
                    quant=cfg.quant,
                    param_count_b=cfg.param_count_b,
                    prompt_name=prompt_name,
                    status="error",
                    error_message=str(e),
                )
            )

    del model
    gc.collect()
    if not nvml_available():
        log(f"[{cfg.label}] warning: pynvml unavailable, GPU util not sampled")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results
