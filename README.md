# Ramer

Two small command-line tools that make local Ollama inference on a
single Windows machine faster and more reliable, plus the research that
led to them:

- **`scripts/ramer_local.py`** -- finds the GPU/CPU layer split for a
  given model that actually maximizes throughput on your hardware
  (Ollama's own default is deliberately conservative and leaves real
  performance on the table), caches it, and watches for it going stale.
- **`scripts/mini_agent.py`** -- a minimal Cline-style coding agent (read
  file / write file / list dir / run command, tool-calling via
  `/api/chat`) that runs on top of a Ramer-optimized placement instead of
  Ollama's default.

## Status

This started as a 10-phase research project on tiered/remote memory for
local LLM inference (`docs/ARCHITECTURE.md`). That research is complete
and closed out -- see its "Status" section for the full arc, including
what didn't pan out. What's left and actively maintained is the narrower,
validated part: **the layer-placement optimizer and the agent built on
top of it.** Every claim below is a real measurement on one real machine
(RTX 5060, 8GB VRAM), not a benchmark suite or a general guarantee.
Full experiment writeups, methodology, and honest negative results
(including a corrected earlier conclusion) are in
`PRODUCT_HYPOTHESES.md`.

This is a personal tool, not a product. No customer discovery has
happened yet -- see `PRODUCT_HYPOTHESES.md`'s closing section for that
plan if you're curious where this could go next.

## Why

Ollama's automatic layer placement is safe but leaves throughput on the
table -- consistently 16-35% in this project's own measurements, across
context lengths from 2048 to 12288 tokens (`results/expC_context_sweep*`).
Push placement too far the other way, past the GPU's physical VRAM, and
throughput doesn't degrade gracefully -- it can fall off a cliff to a
fraction of baseline. That cliff is real but **not fully deterministic**:
the same overflowing placement collapsed to 1.47 tok/s once and stayed
smooth at 13.9 tok/s another time, depending on conditions outside what
either tool can see in advance. So `ramer_local.py` doesn't just pick a
placement once -- it also has to notice, at runtime, when a cached
placement stops matching reality.

## Quick start

Requires Python 3.10+, a local Ollama install, and a **discrete NVIDIA
GPU** (`nvidia-smi` on PATH). Tested on Windows; the discrete-GPU
assumption should hold on Linux too, but that's untested.

**Not applicable to Apple Silicon Macs.** This tool optimizes the split
between VRAM and system RAM across a PCIe bus -- a discrete-GPU problem.
Unified memory (every M-series Mac) has no such split to optimize, and
`nvidia-smi` doesn't exist there regardless. See `PRODUCT_HYPOTHESES.md`
(H2) for why this matters for who this is actually useful to.

```bash
# find and cache the fastest layer split for a model at your context window
python scripts/ramer_local.py probe qwen2.5:14b-instruct --num-ctx 8192

# run at the cached (or auto-probed) optimal split
python scripts/ramer_local.py run qwen2.5:14b-instruct "explain gradient descent"

# see every cached placement, and the history of real runs (auto vs. Ramer tok/s)
python scripts/ramer_local.py show-cache
python scripts/ramer_local.py show-log
```

```bash
# a minimal coding agent, scoped to one working directory, using the
# Ramer-optimized placement for that model/context automatically
python scripts/mini_agent.py --workdir /path/to/your/project qwen2.5:14b-instruct \
    "read the files here and tell me what this project does"
```

## How the placement is kept honest

- **Context-aware**: a placement is only valid at the context window it
  was measured at (the KV cache competes with model weights for the same
  VRAM), so placements are probed and cached per `--num-ctx`, not just
  per model.
- **Headroom-aware**: among candidates that tie on throughput, the
  smallest VRAM footprint wins -- a placement that's 0.04 tok/s faster
  but has zero headroom is a worse bet than one with margin to spare.
- **Self-healing**: every `run` does a cheap pre-flight throughput check
  against the cached number; if it's regressed, that run falls back to
  Ollama's own safe auto-placement and the cache is flagged for a fresh
  probe next time -- no permanent manual `--force` needed. A live
  response is also watched for drift *during* generation, since a short
  pre-flight check can't see a rate that only degrades partway through a
  long response.
- **Deliberately not built**: interrupting or reloading a model
  *mid-generation* in response to detected drift. That's a bigger,
  riskier intervention than anything here does today -- see
  `PRODUCT_HYPOTHESES.md` for why it's parked, not forgotten.

## What this doesn't do

- **MoE expert-level placement.** Mixture-of-experts models (e.g.
  Qwen3.6-35B-A3B) have a finer-grained placement lever -- keep shared
  attention layers on GPU, route individual experts to CPU -- that
  `llama.cpp` supports directly but Ollama's API doesn't expose. Confirmed
  by direct measurement, not assumption; see `PRODUCT_HYPOTHESES.md`.
  Would require driving a different backend, not tuning this tool harder.
- **Anything beyond Ollama.** Both tools speak Ollama's REST API only.

## Repo layout

- `scripts/ramer_local.py`, `scripts/mini_agent.py` -- the two tools.
- `PRODUCT_HYPOTHESES.md` -- every experiment, finding, correction, and
  open question, in the order they actually happened.
- `docs/ARCHITECTURE.md` -- the original 10-phase memory-tiering research
  that preceded and motivated this narrower tool.
- `results/` -- raw JSON/JSONL from the controlled experiments.
- `src/` -- the original research prototype's code (memory-tiering
  simulation harness); not part of the maintained tools above.

## License

MIT, see `LICENSE`.
