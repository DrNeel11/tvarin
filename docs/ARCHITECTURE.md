# Adaptive Memory Tiering for Local LLM Inference: An Experimental Study of Remote Memory, Prefetching, and Persistent Caching

*(Originally scoped as "AI-Powered Remote Memory System for Local LLM
Inference" — renamed after Phase 6 because that title described the
starting hypothesis, not what the experiments actually found. Prediction-
based prefetching, the mechanism the original title centered, turned out
not to be the lever that worked; persistent caching of immutable weights
and the working-set capacity boundary it's bounded by are.)*

## Executive summary (read this first)

**Final thesis**, earned across ten phases, not assumed at the start:

> Remote memory for LLM inference is not fundamentally a prediction
> problem. It is a memory-orchestration problem involving caching,
> workload scheduling, and appropriately granular prefetching. Prediction
> can improve orchestration when workload uncertainty and available hiding
> time justify speculative action, but its benefit is bounded by the cost
> of incorrect decisions and by information that cannot be known in
> advance.

The research narrative, each phase narrowing the previous one rather than
building toward a predetermined answer:

1. Remote memory can extend capacity, but naive access is painfully slow.
2. Network latency/bandwidth determine when remote access is viable at all.
3-4. Fine-grained (per-layer) prefetching cannot hide the transfer cost --
   the compute window is ~100x too small, at any batch size tested.
5. Persistent caching of immutable weights dramatically improves
   performance -- but only up to a hard working-set capacity cliff.
6-7. Batching improves *aggregate* multi-tenant throughput, not individual-
   user latency -- a different mechanism (amortization) than hiding.
8. A shared cache beats a fixed partition when tenants have unequal
   working sets; naive equal splits create guaranteed losers.
8 (coda). Tenant-level prefetch *does* work, because a whole tenant's turn
   provides a large enough window to actually hide a transfer behind.
9. The mechanism generalizes across at least two decoder architecture
   families through a small introspection helper, not a large rewrite.
10. Under a changing workload, prediction needs a cost-aware action policy
   paired with it -- and even the best policy tested still can't eliminate
   the one miss no online system can avoid: the first request after a
   regime change, which looks identical to business as usual until it
   happens.

**The strongest baseline in the whole project is reactive shared-LRU caching
with no prediction at all** -- it outperformed every predictive policy
tested under a changing multi-tenant workload (Phase 10). That is reported
plainly rather than buried, because a research project that can say "the
simple baseline won" is more credible, not less.

If this became a product, the technical hypothesis has now been
investigated thoroughly enough that the next step is not "keep adding ML
until a graph goes up" -- it's establishing whether a real-world workload
exists where this memory orchestration provides an economic advantage over
just buying more RAM.

## Status: research phase complete (2026-09-03)

Tagged `v0.1-research-complete`. Everything below and in `PRODUCT_HYPOTHESES.md`
is preserved as-is; no further technical experiments are planned until
customer discovery produces a specific reason to run one. The arc that
closes this phase:

```
Original hypothesis: "memory offloading is fundamentally broken"
        -> disproven
HF Transformers' apparent cliff
        -> mostly a framework artifact (Experiment A: llama.cpp is ~15x
           more efficient at comparable offload severity)
llama.cpp / Ollama's own CPU<->GPU offload
        -> smooth and monotonic within physical VRAM (Experiment B)
BUT a real cliff exists exactly at the physical VRAM boundary
        -> caused by an unmanaged driver-level fallback, not the engine
Ollama's own conservative auto-placement
        -> leaves real throughput unclaimed, but the fix is a small,
           bounded, upstream-shippable improvement, not a company
```

Reframed conclusion: **the useful problem was never simply "lack of
memory." Modern inference engines already perform substantial CPU/GPU
offloading. The remaining opportunities depend heavily on workload, memory
architecture, and the specific inference engine** -- a materially smaller
and more defensible claim than the project's starting framing.

Next milestone is explicitly non-technical: ~20 customer-discovery
interviews with real local/edge-AI users, asking about their actual
workflow (hardware, models, what happens when something doesn't fit, what
they do about it, what's most frustrating) rather than pitching Ramer or
asking if they'd use it -- see `PRODUCT_HYPOTHESES.md`'s "How to use this
document" section for the question set. Only after a recurring,
unprompted pain pattern emerges from those conversations does any further
building decision get made.

## Competitive landscape (checked, not assumed)

*A significant limitation of this whole document's absolute numbers surfaced
during product validation, worth flagging here even though the full
write-up lives in `PRODUCT_HYPOTHESES.md`: measuring Ollama (llama.cpp)
directly on this same machine at a comparable ~50% CPU/GPU offload split
got 16.43 tok/s, versus Phase 1's 1.1 tok/s under HF Transformers at
similar offload severity -- a ~15x gap attributable to inference-engine
efficiency, not anything this project's tiering/caching work addresses.
The relative findings across Phases 1-10 (the shapes of the effects: the
working-set cliff, batching's amortization-not-hiding mechanism, the
prefetch granularity result) are still sound -- they're internally
consistent comparisons on one framework -- but the absolute "27 -> 1.1
tok/s catastrophe" framing that motivated the project's urgency does not
transfer to a production-grade engine, and shouldn't be quoted as if it
does outside this research context.*

Before any product framing, the load-bearing question: does Phase 8's core
mechanism (a shared GPU memory budget across multiple concurrent models,
LRU-style eviction, predictive prefetching) already exist? Checked
directly rather than assumed either way.

**Prism** (Yu et al., OSDI'26, 20 co-authors, submitted May 2025) already
does close to exactly this. Its open-source component, `kvcached`
(github.com/ovg-project/kvcached), implements a shared memory budget for
model weights across concurrent tenants, LRU-style eviction when a needed
model isn't GPU-resident, and prefetching driven by observed request
patterns -- and the paper's own motivation explicitly studies "a dynamic
bursty-group pattern in which sets of models become active together and
shift over time," the same premise Phase 10 tested independently. It is
**deployed in production across 10,000+ GPUs** and reports 3.3x higher
throughput / 2x better SLO attainment over prior approaches. This is not
"someone might build this" -- it is already built, peer-reviewed, open
source, and running at a scale this project cannot realistically out-build
head-on.

Separately, **LMCache** covers a different but adjacent problem: tiered
*KV-cache* (attention state, not model weights) across CPU RAM, disk, and
remote/object-storage backends, for reuse across requests with shared
prompt prefixes. Orthogonal data structure to everything this project
tiered (weights), but covers "remote-tier caching for LLM serving" broadly
enough that it's a second answer to "why not use an existing tool."

**What survives this check as an actual gap**, not because nothing else is
close, but because of where these two specifically don't reach:

1. **Target hardware.** kvcached's docs and examples are datacenter-framed
   (A100-80G, "organizations," cluster GPU utilization) -- not "one
   developer's 8GB consumer GPU," this project's original framing. Real
   gap, unclear how large a market.
2. **A genuinely networked remote tier.** kvcached moves data only within
   one machine (GPU <-> CPU RAM); it has no cross-machine network fetch.
   Phase 2-3's remote-memory-server work does. But this project's *own
   evidence* argues this specific extension isn't worth much standalone:
   Phase 3/4 already showed naive network-remote fetching collapses
   throughput badly, and the only way it became viable was Phase 5's
   caching and Phase 8's sharing -- exactly the mechanism kvcached already
   provides for local RAM. A genuinely-remote tier mostly matters for the
   narrower case where local VRAM+RAM together still aren't enough, not as
   a general "remote memory for everyone" pitch.

Honest bottom line for the product question: the specific mechanism this
project's research most directly validates (Phase 8's shared adaptive
cache, Phase 8 coda's tenant-aware prefetch) is not an open gap in the
market -- it's already solved, at a more sophisticated level, by a
well-resourced team, in production. Any product angle here has to be about
*who* (consumer/single-machine, not cluster) or a *narrower case*
(genuinely RAM-insufficient workloads) that Prism/kvcached don't target,
not about the core mechanism being unaddressed. That's a much smaller and
more specific claim than the original product framing, and it's one that
needs the customer-discovery step (who actually has this narrower problem,
and would they pay for it) before any more building -- exactly the "find
the wedge" step already identified, now with a concrete reason why the
wedge has to be narrow.

## Research question

Can model-aware AI prefetching and memory scheduling make remote/cloud memory
practical enough to let a resource-constrained PC run larger local LLMs,
without buying more RAM/VRAM? Answer honestly, including "no, except under
conditions X."

## Target machine (measured, not assumed)

- GPU: NVIDIA RTX 5060, 8 GB VRAM, Blackwell (sm_120)
- RAM: 32 GB
- CPU: AMD Ryzen 5 7600, 6C/12T
- Disk: D:\ 474 GB free (NVMe assumed, to be confirmed with a raw fio-style read test)
- OS: Windows 11

## Framework decision

**PyTorch + Hugging Face Transformers + safetensors** for the research
prototype (Phases 1-5). Not llama.cpp initially.

Why:
- safetensors headers separate tensor metadata from data; a single named
  tensor can be read from a memory-mapped file (or substituted with a network
  fetch) without touching the rest of the checkpoint. This is the natural
  interception point for tier-aware loading.
- Transformers exposes `forward_pre_hook` / `forward_hook` per decoder layer
  module — the mechanism for "block until Layer N is resident" and "kick off
  prefetch of N+1 after N finishes."
- llama.cpp already does GPU/CPU layer splitting (`--n-gpu-layers`) and is
  faster in production, but dynamic, per-token, runtime tier decisions aren't
  reachable from Python without patching its C++ internals. Candidate port
  target after the prototype proves the idea, not before.
- Prior art to be aware of and benchmark against, not reinvent: HF
  `accelerate` `dispatch_model(device_map=..., offload_folder=...)` already
  does synchronous CPU/disk offload per layer, but it is not predictive —
  no overlap of transfer with the *previous* layer's compute. FlexGen and
  AirLLM do similar layer-streaming offload for large models on small GPUs.
  Our contribution is (a) a remote tier and (b) predictive/overlapped
  prefetch, measured honestly against these baselines.

## Interception points

- **Load-time**: `safetensors.safe_open(file).get_tensor(name)` — fetch one
  tensor on demand, from local disk or a network call, behind the same
  interface.
- **Runtime**: `register_forward_pre_hook` on each decoder layer (block until
  resident) and `register_forward_hook` (post) to trigger prefetch/eviction
  decisions for the next layer.

## What user-space can control on Windows (v1 constraint: no kernel driver)

Can control: CUDA memory allocation and streams (separate H2D-copy stream
from the compute stream for real overlap), pinned host memory for faster
PCIe transfer, memory-mapped file I/O, TCP sockets, process/thread priority,
`torch.cuda` / `pynvml` telemetry.

Cannot / should not rely on: GPU driver's own WDDM paging, kernel virtual
memory eviction policy. This is why we do explicit tiering rather than trust
`cudaMallocManaged` oversubscription (worth one comparison benchmark later,
but it's coarse and driver-controlled, not something we can make
prediction-aware).

## Benchmark methodology (Phase 1)

- Fixed prompt set (short/medium/long), fixed `max_new_tokens`.
- First run per config discarded as warmup; N=5 timed repetitions; report
  median and note variance (not just mean).
- Per-token timestamps captured via a custom `BaseStreamer` — token
  push-back happens after each decode step, giving TTFT (time to first
  generated token, i.e. prefill latency) separately from steady-state decode
  tokens/sec.
- `torch.cuda.synchronize()` only at measurement boundaries (run start/end),
  never mid-generation — mid-generation syncs would hide the exact
  compute/transfer overlap this whole project exists to study in later
  phases.
- Memory: VRAM via `torch.cuda.max_memory_allocated()` /
  `max_memory_reserved()` (reset peak stats between runs); RAM via
  `psutil.Process().memory_info().rss`; GPU utilization via `pynvml`
  (`nvidia-ml-py` package) sampled on a background thread during generation.
- OOM handling: catch `torch.OutOfMemoryError`, record status="oom",
  `torch.cuda.empty_cache()`, continue to next config — a model that doesn't
  fit is itself a Phase 1 result, not a crash.

## Model ladder (Phase 1)

Qwen2.5-Instruct family: 0.5B, 1.5B, 3B, 7B, 14B. Ungated on HF, permissive
license, consistent architecture across sizes (isolates "size" as the
variable). Each tested at fp16 and, where fp16 doesn't fit 8GB VRAM, 4-bit
(bitsandbytes nf4) as well — this directly produces the
"how large can this machine run entirely locally, and at what quant level"
answer Phase 1 is required to establish.

## Known blockers going in

- RTX 5060 is Blackwell (sm_120) — must verify installed PyTorch build
  actually targets it before trusting any number from it.
- CUDA ops are async; naive wall-clock timing without correctly placed
  syncs measures nothing real.
- Windows has no `tc netem` equivalent — Phase 2 network latency simulation
  will be software-injected delay inside our own remote-memory server, not
  OS-level traffic shaping. This will be stated as a limitation in results,
  not presented as equivalent to real WAN jitter.
- `bitsandbytes` on Windows needs a recent version with prebuilt Windows
  wheels — verify, don't assume it installs cleanly.
- 14B+ at fp16 (~28GB) likely won't fit in 32GB RAM alongside the OS and
  everything else — expected; that boundary is itself a Phase 1 finding.

## Phase status

- [x] Phase 1 — baseline local benchmark (this doc's companion code under
      `src/memtier/benchmark/`, entrypoint `scripts/run_baseline.py`)
- [x] Phase 2 — remote memory server + latency/bandwidth simulation
      (`src/memtier/remote/`, entrypoints `scripts/bench_remote.py` /
      `src/memtier/remote/server.py`)
- [x] Phase 3 — model-aware (layer-level) memory management
      (`src/memtier/tiered/`, entrypoint `scripts/run_tiered.py`)
- [x] Phase 4 (partial) — deterministic "always prefetch next layer" baseline
      implemented and measured; see negative result below. ML-based
      predictor not started.
- [x] Phase 5 (partial) — persistent RAM caching of fetched layers
      implemented and measured (major win, see below), plus a bounded-budget
      / real-eviction test (sharp cliff, see below). Deeper multi-layer-ahead
      pipelining not yet tested. Attempted extending this to the real 14B
      model — blocked by an unresolved segfault, see write-up below.
- [x] Phase 6 — strategy comparison (A-E) and final writeup (below)
- [x] Phase 7 (unplanned) — does batch size change the prefetch picture?
      Tested directly; see write-up below. Short answer: prefetch still
      doesn't help even at batch=128, but batching itself (a different
      mechanism, amortization not overlap) is a large effect -- with an
      important caveat about aggregate vs. per-user throughput.
- [x] Phase 8 (unplanned) — does multi-tenant scheduling actually help,
      per Phase 7's reframed question? Tested directly with two real
      models competing for one cache budget; see write-up below. Short
      answer: yes, clearly -- a shared/adaptive cache beat a fixed 50/50
      split, taking the smaller tenant's throughput from unusable (0.8
      tok/s) to serviceable (~9 tok/s) at zero cost to the other tenant.

## Phase 1 results (this machine, RTX 5060 8GB / 32GB RAM / Ryzen 5 7600)

Full data: `results/phase1_consolidated.csv`. Qwen2.5-Instruct ladder,
fixed short/medium/long prompts, N=5 median unless noted.

| Config | VRAM peak | Decode tok/s | Notes |
|---|---|---|---|
| 0.5B bf16 | 1.0GB | ~30-40 | fully GPU-resident |
| 1.5B bf16 | 3.1GB | ~31-33 | fully GPU-resident |
| 3B bf16 | 6.2GB | ~24-26 | fully GPU-resident |
| 7B bf16 | 7.3GB | **~1.1** | partial CPU offload (exceeds 7GiB cap) — 25x slower than nf4 |
| 7B nf4 | 5.6-5.7GB | ~27-28 | fully GPU-resident |
| 14B bf16 | 7.0GB (+disk) | **~0.0-0.3** | disk+CPU+GPU split; n=1 (reduced fidelity, see below) |
| 14B nf4 | — | — | **load error**: ~7.5GB quantized weights exceed ~7GB free VRAM; bitsandbytes 4-bit has no turnkey CPU-offload path under `device_map="auto"` |

Key findings:

1. **Eager HF `generate()` is dispatch-bound, not compute-bound, for small
   models on this hardware.** GPU utilization during 0.5B decode was only
   ~20-30%, confirming Python/kernel-launch overhead — not raw FLOPs —
   caps small-model throughput. This is a property of the framework choice,
   worth remembering when interpreting all later-phase overlap experiments:
   part of "compute time per layer" here is dispatch overhead, which may
   make prefetch overlap *easier* to hide behind (more slack) or *harder*
   to benefit from (less real compute to overlap with) depending on which
   dominates at a given model size.
2. **The CPU-offload cliff is severe and immediate.** 7B fp16 needs ~14GB
   and only partially fits our 7GiB VRAM cap; the resulting CPU spillover
   dropped decode throughput from ~27 tok/s (7B nf4, fully resident) to
   ~1.1 tok/s — a ~25x penalty from *naive, unscheduled* CPU offload alone,
   before any network tier is involved. This is the number Phase 6's
   AI-scheduled prefetching has to beat.
3. **14B is past this machine's practical local ceiling at any tested
   precision.** fp16 requires disk+CPU+GPU three-way split (~0-0.3 tok/s,
   unusable); nf4 doesn't even load cleanly, since its ~7.5GB of quantized
   weights exceed the ~7GB truly free on an 8GB card once other GPU
   consumers are accounted for, and bitsandbytes' CPU-offload fallback
   (`llm_int8_enable_fp32_cpu_offload`) failed under `device_map="auto"`
   with a meta-tensor error — it would need a hand-built device map to work
   at all, which was out of scope for a Phase 1 baseline. **This is Phase
   1's headline answer**: on this machine, "can it run locally at all" and
   "can it run *usably*" both say no somewhere between 7B and 14B, which is
   exactly the gap Phases 3-6 are testing whether remote-memory prefetching
   can close.
4. Download bandwidth (~8.7MB/s sustained from the unauthenticated HF Hub
   on this connection), not GPU compute, was the dominant wall-clock cost
   of Phase 1 itself — unrelated to the research question but worth noting
   for reproducing this experiment.
5. The 14B-fp16 row is N=1, no warmup (downgraded from the standard N=5
   protocol after estimating the full run would take hours at <1 tok/s);
   its "short" prompt TTFT (173s) is a cold-disk-cache outlier from
   skipping warmup, not representative — medium/long (~3.8s TTFT once
   warm) are more indicative. Treat this row as anecdotal, not a robust
   median, consistent with the "no fake benchmark results" requirement.

## Phase 2 results (remote-link simulation, standalone, no model involved)

Full data: latest `results/*_phase2_remote_link/results.csv`.

- Bandwidth throttling is accurate: measured throughput matched the
  configured cap within ~1% (e.g. 10MB/s cap -> 9.9-9.96MB/s measured,
  2MB/s cap -> 1.96-2.0MB/s measured).
- Small blocks are latency-dominated: a 1MB block's effective throughput
  fell from 465 to 20 MB/s as injected latency rose from 1ms to 50ms.
- Large blocks are bandwidth-dominated: a 256MB block got ~1100-1400MB/s
  effective throughput across the *entire* 1-50ms latency range —
  latency stopped mattering once transfer time dominated.
- This crossover (small=latency-bound, large=bandwidth-bound) is exactly
  what Phase 3+ needs to reason about when deciding whether prefetching a
  given tensor size at a given simulated network condition is worth it.

## Phase 3 & 4 results (layer-level tiering, synchronous vs. prefetch)

Qwen2.5-1.5B (28 layers, ~95MB/layer bf16), last 4 layers forced onto the
"remote" tier (evicted after every use, so every decode step re-fetches
them), 10ms latency / 1000MB/s bandwidth:

| Strategy | Decode tok/s | vs. fully-local baseline (~31 tok/s) |
|---|---|---|
| Fully local (Phase 1) | ~31 | baseline |
| Synchronous fetch-on-demand (Phase 3) | 1.6 | ~19x slower |
| + deterministic prefetch-next-layer (Phase 4) | 1.5-1.6 | **no improvement** |

**Prefetching the next layer while the current one computes did not help,
and the reason is structural, not a bug**: per-layer compute time at
batch=1 decode is ~1-2ms (33ms/token ÷ 28 layers), but fetching a ~95MB
layer took ~140-155ms even at a generous 1000MB/s / 10ms link. Instrumented
per-fetch: 2295/3060 fetches did have a prefetch in flight when needed, but
*zero* were fully hidden (wait < 1ms) — the compute window available to
prefetch behind was ~1% the size of the transfer it needed to hide.

This generalizes past this one model: at batch size 1, both compute time
and transfer time per layer scale roughly with the same quantity (layer
parameter count) for memory-bandwidth-bound matvec ops, so the
compute-time-to-transfer-time ratio is roughly *independent of model size*
-- a 7B or 14B model's per-layer compute window is proportionally larger,
but so is its per-layer transfer requirement. One-layer-ahead prefetch
alone cannot close a gap that exists at the architecture level for
single-stream decoding. Closing it would need either (a) a network tier
approaching local RAM bandwidth (at which point it's arguably not "remote"
memory anymore), (b) prefetching further ahead than one layer to build a
deeper pipeline, or (c) larger batch sizes, where compute scales with batch
size while weight transfer (reused across the batch) does not, improving
arithmetic intensity in prefetching's favor. (c) was not tested here — this
prototype is batch=1 throughout, matching single-user local inference,
but it is the most promising direction to revisit if Phase 5/6 want a
scenario where scheduling can actually win.

This is exactly the kind of honest negative result the project brief asked
not to paper over: naive "always prefetch next layer" is not sufficient to
make remote memory practical for single-stream decoding on this hardware.
Phase 5's smarter scheduling has a real, structural gap to try to close,
not just an implementation inefficiency in Phase 4's baseline.

## Phase 5 result: persistent RAM caching (the actual win)

Prefetch-ahead failed because it tried to hide a *recurring* per-token
network cost behind a compute window ~100x too small. But the network cost
was never structurally required to recur: model weights are static and
never change between tokens, so re-fetching the same 95MB layer from the
network on every single decode step (Phase 3/4's behavior) was pure waste.

Same config as Phase 3/4 (4/28 layers on "remote", 10ms/1000MB/s), but the
tier manager now keeps a fetched layer resident in an LRU-bounded RAM cache
instead of evicting it after use -- only the transient GPU copy is freed
each step, not the RAM copy:

| Strategy | Decode tok/s | vs. fully-local (~31) |
|---|---|---|
| Synchronous fetch-on-demand (Phase 3) | 1.6 | ~19x slower |
| + prefetch next layer (Phase 4) | 1.5-1.6 | no improvement |
| + persistent RAM cache (Phase 5) | **15-16** | ~2x slower |

RAM cache hit rate: 3056/3060 fetches (99.9%) -- only the 4 unavoidable
first-touch cold fetches (one per overflow layer, for the whole run) missed
and paid the network cost; every subsequent token's access to those same 4
layers was served from RAM.

The remaining ~2x gap to the fully-local baseline is no longer a network
cost at all -- avg fetch wall time dropped from ~154ms to ~8.4ms, and that
residual 8.4ms is PCIe host-to-device copy + Python dispatch overhead for
re-uploading the cached layer to GPU every token (still necessary because
these 4 layers structurally don't fit in the VRAM budget, so they still
can't stay GPU-resident). That's a RAM-tier cost, not a remote-tier cost --
exactly the distinction the architecture doc's tier diagram draws, now
verified empirically rather than assumed. Closing the remaining gap would
mean overlapping that H2D copy with compute via a dedicated CUDA copy
stream (deeper engineering, not yet done) rather than anything to do with
the network simulation.

**Practical implication for Phase 6**: for a model whose overflow layers
fit within combined VRAM+RAM (this experiment, and Phase 1's 7B-on-32GB-RAM
case), a scheduler that (a) caches static weights in RAM after first fetch
and (b) doesn't bother prefetching-ahead for layers that will hit cache
anyway gets most of the achievable win without needing deep pipelining or
ML-based prediction. Prefetch-ahead's real value should reshow itself
specifically for the *first, unavoidable* cold-fetch case (hiding that
initial network latency behind whatever compute precedes it) and for
scenarios where RAM itself is the constraint.

## Phase 5 follow-up: bounded RAM budget is a cliff, not a slope

The 15-16 tok/s result used an *unbounded* RAM cache -- room for all 4
overflow layers (~380MB) at once. What happens when RAM itself can't hold
the full working set, closer to Phase 1's 14B case where even RAM
overflows? Same config, cache capped to fit only 3 of the 4 layers
(285MB budget):

| RAM cache budget | Decode tok/s | Cache hit rate |
|---|---|---|
| Unbounded (fits all 4 layers) | 15-16 | 99.9% (3056/3060) |
| 3 of 4 layers' worth | **1.6** | **0.0% (0/3060)** |

Not a graceful degradation -- a cliff. The access pattern here is a strict
cycle (layers A,B,C,D,A,B,C,D,... every decode step, same order, forever).
Under a cache smaller than one full cycle, *every* eviction removes the
exact item that will be needed next, before any currently-cached item is
touched again -- so the hit rate is 0% regardless of eviction policy, not
just for LRU. (This is a known cache-theory result: cyclic/sequential scans
longer than cache capacity are worst-case for essentially any practical
replacement policy, including Belady's own optimal offline algorithm.)

The actionable consequence for Phase 6's scheduler: **once memory is
genuinely insufficient, tuning *which* layer to evict is pointless for this
access pattern -- the only lever that matters is the binary decision of
*how many whole layers* to commit to a tier**, decided once at placement
time (Phase 3's problem), not adjusted continuously by an eviction policy
(Phase 5's original framing). A smarter scheduler's job here isn't "evict
better" -- it's "know in advance whether a tier's budget covers the full
working set, and if not, don't bother caching partially; fall back to
synchronous fetch (or don't offload that many layers to begin with)."

## Attempted: running the actual 14B model through the tier manager

All Phase 3-5 results above use Qwen2.5-1.5B with a *synthetic* overflow
(4 of its 28 layers artificially forced onto the remote tier). The natural
next question -- does the same mechanism (persistent RAM caching, the
working-set cliff) hold on the real 14B model that motivated the project,
where Phase 1 saw accelerate's own opaque offload collapse to ~0-0.3 tok/s?

Approach: reuse the proven `device_map="auto"` + `max_memory` loading path
from Phase 1 (not a hand-rolled meta-tensor loader, to avoid correctness
risk around buffers like rotary embeddings), but with a larger CPU budget
(26GiB vs Phase 1's 22GiB) so accelerate's own placement doesn't need disk
at all. This worked cleanly: accelerate placed 7/48 layers on GPU, 41 on
CPU, no disk, confirmed via `model.hf_device_map` (see
`src/memtier/models/registry.py`'s `qwen2.5-14b-fp16-tiered` config).

Extracting those 41 CPU-resident layers to hand to our own tier manager
did not work. accelerate's multi-device dispatch represents CPU-offloaded
layers as meta-tensor placeholders backed by its own hooks -- reading them
via `layer.named_parameters()` fails ("Cannot copy out of meta tensor").
The fix -- detach accelerate's hooks (`accelerate.hooks.remove_hook_from_
module`) and read the same layers' weights directly from the safetensors
checkpoint on disk instead (`src/memtier/tiered/safetensors_reader.py`,
bypassing the live model's representation entirely) -- got past that error,
but **the process then segfaulted** (a hard native crash, not a Python
exception) while materializing the ~22.6GB of CPU-tier layer data. A
suspected cause (reading all 41 layers into a staging dict before handing
each to the manager, double-buffering every layer's weights in memory at
once) was fixed by switching to a lazy, one-layer-at-a-time loader --
the segfault recurred anyway, with the crash point moving *earlier* (before
the per-layer reading code even executes), suggesting the real cause is
something deeper than the staging-buffer hypothesis -- plausibly repeated
large-model loads destabilizing the process, or a lower-level conflict
between accelerate's own memory-mapped file handles on the checkpoint
shards and a second, independent `safetensors.safe_open` session on the
same files. Not resolved within this session's scope.

System safety was verified after each crash (17-18GB RAM free, no
orphaned processes, GPU VRAM back to baseline) -- the crash was contained
to the one process each time, not a system-stability risk, just an
unresolved bug in this specific loading path.

**Consequence**: the working-set-cliff and RAM-caching mechanism are
verified on Qwen2.5-1.5B with a synthetic overflow, and Phase 1 separately
measured the real 14B model's collapse under accelerate's own (non-tiered)
offload -- but the two haven't been connected with our own instrumented
tier manager. Revisiting this would need either a from-scratch safetensors
loader (skip `device_map="auto"`/accelerate's dispatch entirely, avoiding
its meta-tensor+hook representation from the start, at the cost of needing
to handle buffers like rotary embeddings correctly by hand) or deeper
debugging of the segfault itself (e.g. reproducing it in isolation outside
the full pipeline, checking for an accelerate/safetensors mmap interaction
bug) -- both beyond what this session attempted.

The 14B segfault is an **engineering limitation of this prototype's loading
path**, not evidence against remote-memory tiering as a concept. It says
nothing about whether the mechanism verified on 1.5B (caching + the
working-set cliff) would or wouldn't hold at 14B scale -- that question
remains genuinely open, not answered negatively.

---

# Phase 6: Strategy Comparison and Final Research Conclusions

## A-E comparison

All numbers from the controlled Qwen2.5-1.5B experiment (28 layers, 4
forced onto a non-GPU tier, 10ms latency / 1000MB/s bandwidth simulated
link) unless marked otherwise. This is the apples-to-apples comparison;
Phase 1's real-hardware, real-model-size numbers (7B, 14B) are corroborating
evidence for the same pattern at a scale this experiment couldn't safely
reach (see the segfault write-up above).

| # | Strategy | Mechanism | Decode tok/s | vs. local |
|---|---|---|---|---|
| A | Normal local inference | Everything GPU-resident | ~31 | baseline |
| B | Naive remote offloading | Fetch on demand, evict after every use (Phase 3) | 1.6 | ~19x slower |
| C | Sequential prefetching | Always prefetch layer N+1 during layer N's compute (Phase 4) | 1.5-1.6 | **no improvement over B** |
| D | AI-based prefetching | *Not built* -- see below | -- | -- |
| E | AI-based memory scheduling | *Not ML-based; see below* | 15-16 (unbounded cache) / 1.6 (cache below working set) | ~2x slower / ~19x slower |

**Why D (a learned prefetch predictor) was never built**: C already
isolated the failure mode with certainty. At batch=1, per-layer compute
(~1-2ms) is roughly two orders of magnitude smaller than the transfer time
of a remotely-fetched layer (~140ms+ at a generous simulated link).
Instrumentation showed 2295/3060 fetches had a prefetch already in flight
when needed, and *zero* were fully hidden -- the shortfall is in the
*size of the window available to hide behind*, not in *predicting what to
prefetch*. Layer order in a transformer decode pass is already
deterministic and perfectly known one step ahead; there is no prediction
problem here for a learned model to solve better than the trivial
"always prefetch the next layer" policy already does. Spending effort on
gradient-boosted-tree or neural predictors would have optimized a variable
(*what* to prefetch) that was never the bottleneck (*how much compute-time
there is to hide the prefetch behind*). Building D would have been
motion, not progress -- this is itself a finding, not a gap: **the project's
own instrumentation correctly ruled out an entire planned phase as
unnecessary before investing in it**, which is what the phased,
measure-before-building approach in this doc's methodology is for.

**Why E turned out not to be "AI" in the ML sense**: the effective
scheduling decision that produced the 10x win (row E's 15-16 tok/s) is not
a learned policy -- it is the simple, architecturally-motivated observation
that model weights are static, so a fetched layer should be cached and
reused rather than evicted and re-fetched. No model, no training data, no
inference-time prediction step. The *harder* scheduling decision this
project surfaced -- the working-set cliff (row E's other number, 1.6 tok/s
when the cache is undersized) -- was shown to be **unimprovable by any
eviction policy** for this workload's cyclic access pattern, including
Belady's theoretically optimal offline algorithm. An "AI-based" eviction
policy would have hit the identical wall a rule-based one does, because the
ceiling is set by the access pattern and the memory budget, not by policy
quality. Where intelligence remains genuinely useful is upstream of
eviction: the *placement* decision of how many whole layers a tier's
budget can support before commitment -- a capacity-planning problem, not a
prediction problem.

## Answering the research question

> Can model-aware AI prefetching and memory scheduling make remote/cloud
> memory practical enough to allow resource-constrained PCs to run larger
> local LLMs without additional physical RAM/VRAM?

**The honest answer, as measured, is: not primarily through prefetching or
learned scheduling -- but a related and simpler mechanism gets most of the
way there, with a hard boundary the project could not fully map.**

Specifically:

1. **Naive remote memory is not practical for single-stream (batch=1)
   decoding.** ~19x slowdown, confirmed both in the controlled 1.5B
   experiment and in Phase 1's real-hardware 7B/14B numbers under
   accelerate's own (unrelated, independently-implemented) offload.
2. **Prediction-based prefetching does not fix this**, and the reason is
   structural (a batch=1-independent mismatch between per-layer compute
   time and per-layer transfer time), not a matter of better predictions --
   this generalizes past the one model size tested (Phase 3/4's writeup
   above explains why the compute/transfer ratio is roughly model-size-
   invariant at batch=1).
3. **What works is exploiting a property of the workload the original
   hypothesis didn't center**: weights are immutable across the whole
   generation. Caching them after first fetch, rather than re-fetching
   every token, recovered ~10x of the lost throughput (1.6 -> 15-16
   tok/s) using no prediction or learning at all.
4. **That mechanism has a hard ceiling**, not a graceful degradation: it
   requires the *entire* recurring working set to fit in the cache tier.
   One layer short of that, under this cyclic access pattern, the win
   disappears completely (back to ~1.6 tok/s) regardless of eviction
   policy sophistication.
5. **Where that ceiling actually sits for a model that doesn't fit in
   combined VRAM+RAM at all (this project's original motivating case, e.g.
   14B on this 8GB/32GB machine) was not established** -- the attempt hit
   an unresolved engineering bug (segfault), not a research finding. Phase
   1's independent, non-tiered measurement of 14B (~0-0.3 tok/s under
   accelerate's own offload, needing disk) is consistent with 14B being at
   or past that ceiling on this specific machine, but this project did not
   verify that its own caching mechanism would (or wouldn't) do better than
   accelerate's naive offload at that scale.

**Practical framing for the product vision** ("run larger local models
without buying more RAM/VRAM"): the evidence here supports a narrower but
still useful claim than the original hypothesis -- a scheduler that (a)
maximizes what fits in VRAM, (b) caches the rest in RAM rather than
re-fetching it, and (c) only reaches for genuinely remote/network memory
for whatever doesn't fit in (a)+(b), should recover most of the achievable
performance for models whose overflow fits in combined VRAM+RAM. For models
whose overflow *doesn't* fit even in RAM, this project's evidence neither
confirms nor rules out that remote memory can still help -- that remains
the open question, gated on resolving the loading-path bug rather than on
any further prefetching or scheduling cleverness.

## What would come next (not pursued further this session)

- Debug the 14B segfault in isolation, or build a from-scratch safetensors
  loader that bypasses accelerate's dispatch entirely, then re-run the
  Phase 5 cache/cliff experiment at a scale where RAM itself is genuinely
  insufficient (this project's original motivating scenario).
- Test larger batch sizes: Phase 3/4's negative prefetching result was
  specific to batch=1 decode, where compute-per-layer and transfer-per-
  layer scale together. Larger batches increase compute per layer without
  increasing weight-transfer volume (weights are reused across the batch),
  which is the standard lever for improving arithmetic intensity and could
  make prefetch overlap viable where it wasn't here.
- Deeper multi-layer-ahead pipelining (buffer several layers' worth of
  prefetch in flight at once, not just one) as a distinct lever from batch
  size -- untested in this session.

## Where ML would actually be justified (not attempted this session)

Phase 6 showed AI/ML predicting *layer sequence* was never the right
target -- decode order is already deterministic, so there was nothing to
predict better than "next." That doesn't mean ML has no role; it means the
role is different from the one originally scoped. The distinction that
matters:

- **Bad fit**: "which layer comes next" -- already known, zero uncertainty.
- **Good fit**: "which data is worth keeping resident, for how long, under
  which conditions" -- genuinely uncertain, and exactly the kind of
  decision this project's simple rules (cache everything static; evict
  only under pressure) can't make once the workload stops being a single
  deterministic stream.

Concrete cases where that uncertainty shows up, roughly in order of how
directly they follow from this project's findings:

1. **Predicting the required cache size before committing to a placement.**
   Given the working-set cliff (Phase 5), the highest-value prediction
   isn't "what to prefetch" but "will this configuration's cache actually
   cover the recurring working set, or is it about to thrash" -- a
   go/no-go decision made once at placement time, from hardware +
   model + context-length + concurrency features, rather than a per-token
   decision.
2. **Predicting whether a remote fetch is worth starting at all**, i.e. a
   direct, learnable version of the arithmetic this project did by hand in
   Phase 2/3 (transfer time vs. time-until-needed) -- becomes genuinely
   hard to hand-compute once network conditions are non-stationary or
   several tensors compete for the same link.
3. **Multi-tenant / multi-request scheduling.** Every experiment here was a
   single deterministic decode stream. Serving several requests (possibly
   different models) concurrently turns "what stays resident" into a
   real contention problem with no fixed answer -- reuse depends on which
   requests are active, not on layer index. This is the scenario where
   "predict reuse, not sequence" stops being optional.
4. **Adaptive tier selection under changing conditions** (VRAM/RAM/SSD/
   remote, choosing per-tensor rather than per-layer-block) and
   **network-aware pacing** (Phase 2 already showed latency/bandwidth
   trade off differently by transfer size -- a policy that adapts to
   observed conditions rather than a fixed simulated link is a natural
   extension of that measurement, not a new mechanism).

## Phase 7: does batch size change the prefetch picture?

Phase 3/4's negative prefetch result was specific to batch=1. The
hypothesis (stated in Phase 6's writeup): compute-per-layer scales with
batch size while weight-transfer size doesn't (weights are reused across
the batch), so a big enough batch should let compute finally cover the
transfer. Tested directly: same 1.5B / 4-remote-layer / 10ms / 1000MB/s
config as Phase 3/4, `--batch-size` replicating the prompt across the
batch dimension (identical content, so no padding needed -- this measures
aggregate system throughput under repeated/parallel access to the same
offloaded layers, not per-token latency for a single stream).

| batch_size | prefetch | aggregate decode tok/s | fully-hidden fetches |
|---|---|---|---|
| 1 | no | 1.6 | -- |
| 1 | yes | 1.5-1.6 | 0/3060 |
| 8 | no | ~12.0 | -- |
| 8 | yes | ~11.6 | 0/2928 |
| 128 | no | ~195-201 | -- |
| 128 | yes | ~188-190 | 0/3204 |

Two findings, and they cut in different directions:

**Prefetch still doesn't help, even at batch=128.** Avg fetch wall time
stayed ~155-162ms regardless of batch size (expected -- weight size is
batch-independent), and *zero* fetches were fully hidden at any batch size
tested. The reason: per-layer compute time did not grow proportionally
with batch size in this range. This model/GPU/batch combination stays deep
in a memory-bandwidth- and dispatch-overhead-bound regime (consistent with
Phase 1's finding that eager HF `generate()` is dispatch-bound at small
batch) rather than becoming compute-bound -- the fixed per-step dispatch
overhead doesn't scale with batch size at all, so it dominates total step
time even at batch=128, leaving little room for genuine FLOP-driven compute
to grow into. The theoretical "batch size fixes the compute/transfer ratio"
argument from Phase 6 turns out to need a compute-bound regime this
setup never reached; prefetch's negative result generalizes further than
Phase 6 already concluded, not less.

**But batching alone -- no prefetch, no caching, none of this project's
mechanisms -- turned naive synchronous fetching from catastrophic into
excellent, via a completely different mechanism than "hiding": amortization,
not overlap.** The same ~150ms fetch that produces 1 token at batch=1
produces 128 tokens at batch=128, for the same fixed cost. Aggregate
throughput scales roughly with batch size because the fetch cost is shared
across it, not because any individual token got faster.

**The catch, and it's the one that matters for this project's original
framing**: aggregate throughput is not per-user throughput. At batch=128,
~197 tok/s aggregate over 128 simultaneous streams works out to
~197/128 ≈ **1.5 tok/s per individual stream -- statistically the same as
the unbatched, unimproved batch=1 number.** Batching doesn't make any one
user's generation faster; it lets more users share the same fetch cost.
That makes it a genuinely valuable mechanism for a **multi-tenant server**
amortizing one remote-memory link across many concurrent requests (see
the "multi-tenant scheduling" ML opportunity above -- this experiment is
direct empirical support for that being the scenario where the economics
actually work), but it does **not** address this project's original
single-user framing ("run a bigger model on my own machine for my own
use") -- a lone user cannot batch their own single request 128 ways to
make their own chat response arrive faster; they'd need 128 *other*
concurrent requests to share the cost with, which changes the deployment
model, not just the workload.

Architecturally, the natural split is **ML predicts, a scheduler acts**:
a lightweight model (start with logistic regression / gradient boosting,
per the original spec's own guidance -- only escalate if benchmarks show
it's insufficient) outputs a reuse/value estimate per tensor, and the
existing rule-based mechanisms from Phases 3-5 (the tier manager, the
persistent cache, the placement logic) consume that estimate instead of a
fixed static tier_map. None of Phases 1-6's code needs to change for this
-- `LayerTierManager`'s `tier_map` and `cache_after_fetch` policy are
already the seam where a learned policy would plug in, once there's a
workload with real uncertainty to learn from.

## How the research question evolved

The question at the top of this document was the one this project started
with. It is not the one the evidence ends up supporting, and that gap is
the actual result, not a failure to answer the original question.

**Started with**: can AI-driven prediction and prefetching make remote
memory fast enough for *one local user* to run a bigger model than their
VRAM+RAM allow?

**Phases 3-4 answer, directly**: not through prediction or prefetching --
tested at batch=1 (Phase 4) and again up to batch=128 (Phase 7), and in
every case, zero fetches were ever hidden behind compute. This isn't a
tuning failure; per-layer compute time never grew large enough relative to
transfer time in any regime this project's hardware reached, and the
project's own instrumentation is what established that, rather than
assuming it.

**What the evidence supports instead**: two mechanisms that have nothing
to do with predicting what comes next. Persistent caching (Phase 5) helps
enormously when the same weights are reused *over time* by one stream, but
only up to a hard capacity cliff. Batching (Phase 7) helps enormously when
the same weights are reused *across many concurrent streams*, but the
benefit is amortization of a fixed cost, not a faster individual stream --
it is a multi-tenant-serving result, not a single-user one.

**The question this evidence actually supports going forward**: can
intelligent workload scheduling and memory-tier management make
remote/disaggregated memory practical for serving *multiple concurrent*
LLM requests on resource-constrained hardware -- where the optimization
target is maximizing useful work extracted per remote-memory transfer
across requests, not making one request's remote access behave like local
RAM? That reframing is where this project's own AI/ML placement (the
"where ML would actually be justified" section above) already points: not
at predicting deterministic layer sequence, but at predicting reuse,
residency value, and request grouping under genuine uncertainty -- a
scheduling problem across requests, not a prefetch-timing problem within
one.

Phase 8 (below) tested this reframed question directly rather than
leaving it as a proposal: does resource-allocation policy across competing
tenants actually matter, using the simplest possible non-learned policy
comparison. It does, decisively -- see Phase 8's numbers. This project
did not build a learned scheduler. It built the evidence that a scheduler
of *some* kind, allocating a shared resource intelligently across
requests rather than statically dividing it, is the direction actually
supported by measurement -- and a concrete baseline (shared LRU beating a
50/50 split) that any learned policy would need to beat to justify its
own complexity.

## Phase 8: does multi-tenant scheduling actually help?

The reframed question needed a direct test, not just an argument. Two real
models (Qwen2.5-0.5B, 6/24 layers forced remote, ~179MB full working set;
Qwen2.5-1.5B, 8/28 layers forced remote, ~749MB full working set) loaded
simultaneously on the same GPU, sharing one simulated remote link (10ms /
1000MB/s), served with strictly alternating requests (A, B, A, B, ... for
6 rounds) -- two tenants competing for one machine's VRAM+RAM+remote
budget. Combined RAM cache budget fixed at 750MB (~= B's need alone,
comfortably above A's, well below the ~928MB both would need cached
simultaneously) across two allocation policies (`scripts/
run_multi_tenant.py`, `src/memtier/tiered/shared_cache.py`):

| Policy | Tenant A (0.5B) tok/s | Tenant B (1.5B) tok/s | Combined cache hit rate |
|---|---|---|---|
| **Partitioned** (fixed 375MB each) | 13.2-18.8 (steady) | **0.77-0.81 (steady)** | A: 98.1% / B: 0.0% |
| **Shared** (one 750MB pool) | 15.7-19.2 (steady) | **8.6-9.6 (steady)** | 88.9% overall, 46/76 evictions cross-tenant |

Splitting the budget in half looks fair and is exactly what a naive
implementation would do -- and it produces a permanent winner and a
permanent loser: A's 179MB need fits easily in its 375MB half (98.1% hit
rate, consistently fast), but B's 749MB need doesn't fit in its 375MB half
at all (0.0% hit rate -- the same working-set cliff from Phase 5, now
imposed on B by an arbitrary allocation rule rather than genuine hardware
scarcity). B never recovers, round after round.

Sharing one pool removes the artificial cap. Each tenant can use up to the
full 750MB while it's actively running (the other isn't fetching
concurrently in this alternating pattern), so A -- which only ever needed
179MB -- loses nothing, while B finally gets enough room to mostly fit its
749MB need. The cost is visible and real (46 cross-tenant evictions: each
tenant's data does get displaced by the other's, forcing some recurring
cold-refetch at the start of each turn -- which is why B's hit rate is
88.9%, not near-100% the way single-tenant Phase 5 reached with a
sufficient budget), but the net effect is decisively better for the system
as a whole: **B goes from unusable (0.8 tok/s) to actually serviceable
(~9 tok/s) at zero cost to A**, just by removing a fixed partition that was
never load-aware in the first place.

This is a small, deliberately simple policy comparison -- static "split
the number in half" vs. "don't split it, let LRU sort it out" -- not a
learned one. That's the point: it's the cleanest possible demonstration
that the *scheduling decision itself* (how budget is allocated across
competing tenants) matters as much as the caching mechanism Phase 5 found,
and it's exactly the kind of decision -- not layer sequence, but resource
allocation across heterogeneous, competing workloads -- this document's
"where ML would actually be justified" section pointed to. A learned
policy's job here would be sizing partitions (or deciding to not partition
at all) from predicted per-tenant working-set size and request frequency,
rather than assuming equal shares -- the naive-but-correct baseline this
experiment establishes is exactly what such a policy would need to beat.

### Phase 8 coda: prediction-based prefetching, tried again at the right granularity

Phases 3/4/7 killed per-layer prefetching decisively: the compute window
(~1-2ms) was never within two orders of magnitude of the transfer it
needed to hide (~150ms), at any batch size tested. But that conclusion was
specific to *that* granularity. The multi-tenant setup has a different,
much larger window available: while tenant A generates its ~128-token
response (seconds), tenant B does nothing -- an idle window vastly bigger
than the time needed to prefetch B's whole working set (~750ms for 8
layers at this link speed). Same mechanism as Phase 4 (fetch ahead of
need), applied where the timing argument the theory always required
actually holds.

Tested directly: prefetch the *other* tenant's full working set into the
shared cache during each tenant's turn (`LayerTierManager.
prefetch_all_remote_layers()`, one dedicated background connection per
tenant so concurrent prefetches don't corrupt the single-outstanding-
request wire protocol), against the same reactive-shared baseline:

| Scheme | Tenant A tok/s | Tenant B tok/s | Cache hit rate |
|---|---|---|---|
| Shared (reactive) | 15.7-19.2 | 8.6-9.6 | 88.9% |
| Shared + tenant-switch prefetch | ~20-22 (after round 0) | 9.3-11.2 | 88.0% |

A real gain for both tenants (~10-20%), and a telling detail: the hit rate
barely moved (88.0% vs 88.9%) -- essentially the *same* fetches still miss
the cache, but under prefetch those misses happen in the background during
the other tenant's turn instead of blocking the foreground request. The
mechanism is exactly Phase 5's insight (don't pay synchronously for
something you already know you'll need) applied predictively instead of
reactively, at a granularity where "predict what's needed next" is both
easy (round-robin order is simplest case, but a real scheduler would infer
it from request timing/patterns -- the ML opportunity this document
already scoped) and, critically, *useful*, because the window to act in it
is finally the right size.

This is the one place all session where prediction-based prefetching
earned its keep. The through-line across Phases 3-8: prefetching isn't
categorically useless, as Phases 3/4/7 might suggest in isolation -- it's
useless at a granularity where the thing being predicted (next layer) is
both already-known and too fine-grained to matter, and useful at a
granularity where genuine scheduling uncertainty (which tenant, when) and
a large enough window coincide. That distinction -- not "prefetching
works" or "prefetching doesn't work" -- is this project's actual finding
about when prediction has a job to do in a memory-tiering system.

## Phase 9: how much of "model-agnostic" is actually needed?

A natural next milestone once Phase 8 worked: generalize past two known
Qwen models to arbitrary Hugging Face causal LMs. The full version of that
idea is a large undertaking -- a generic memory-object abstraction
decoupling model runtime from tier placement, automatic model
introspection, lazy per-tensor loading, KV-cache-aware placement -- weeks
of engineering, not something to commit to on assumption. Tested the
actual premise first instead, cheaply: does the current code's one
hardcoded assumption (`model.model.layers`) actually break on a different
architecture, and if so, how much does fixing it really require?

Confirmed directly: `model.model.layers` raises `AttributeError` outright
on `EleutherAI/pythia-160m` (`GPTNeoXForCausalLM` has no `.model`
attribute -- the real path is `model.gpt_neox.layers`). Fix: a ~40-line
helper (`src/memtier/tiered/model_introspection.py`) trying a handful of
known naming conventions (Llama/Mistral/Qwen2/Gemma/Phi's `model.layers`;
GPT-NeoX/Pythia's `gpt_neox.layers`; GPT-2/BLOOM's `transformer.h`;
MPT's `transformer.blocks`) and returning whichever matches, with a clear
error naming the actual module tree if none do. Verified end-to-end, not
just at the discovery step: pointed the full Phase 3/5 pipeline (layer
discovery -> VRAM offload -> remote push -> persistent cache -> decode) at
Pythia-160M with 3/12 layers forced remote. VRAM offloading was exactly
right (0.32GB -> 0.28GB), and the tiered decode itself worked (85.25
tok/s on the first prompt) -- the only failure was unrelated, a
non-tiering assumption elsewhere in the benchmark harness (`tokenizer.
apply_chat_template` assumed an instruction-tuned model; Pythia is a base
model with no chat template) fixed with a similarly small guard in
`run_one_prompt`.

**Conclusion, precisely stated**: not "any model works" -- two data points
(Qwen2 and GPT-NeoX) is evidence the approach crosses architecture
families through lightweight introspection, not evidence every
architecture is supported. The defensible claim is narrower: *the tiering
mechanism can operate across at least two distinct decoder architecture
families via lightweight model introspection rather than model-specific
layer handling*. That's still a small and mechanical gap (a handful of
naming conventions to try, one line to extend if a new one is hit), not
evidence for the large generic-object architecture proposed as the way to
close it. That architecture may still be worth building for other reasons
(a real serving system spanning many more exotic architectures, KV-cache
placement, lazy loading of models too large to inspect eagerly), but *this
project's own evidence* -- the same standard this document has held every
other claim to -- supports a much smaller fix for the specific gap
measured, not the full rewrite. Extending `_KNOWN_PATHS` for a genuinely
new naming convention, if one is ever hit, is a one-line addition; nothing
observed so far justifies more than that.

## Phase 10: does prediction quality matter under a changing workload?

Phase 8's coda showed tenant-switch prefetch working, but it assumed a
known, fixed A/B/A/B schedule and always prefetched "the other tenant" --
true by construction there. Real workloads shift. Tested with a 16-round
schedule that alternates, then bursts into 4 rounds of only A, then 4
rounds of only B, then alternates again (`scripts/
run_multi_tenant_dynamic.py`), comparing three policies at the same 750MB
shared budget:

| Scheme | Hit rate | Evictions | Cross-tenant evictions |
|---|---|---|---|
| Reactive (no prefetch) | **93.1%** | 62 | 38 |
| Naive (always assume alternation, Phase 8 coda's policy unchanged) | 87.9% | 115 | 62 |
| Adaptive (online first-order Markov, recency-windowed) | 90.2% | 93 | 46 |

**First result**: the naive fixed-pattern policy that worked in Phase 8
actively hurts once its assumption stops holding. During each burst it
keeps confidently prefetching the tenant that *isn't* coming next,
evicting the correct tenant's warm cache for no benefit -- nearly double
the evictions of doing nothing at all.

**Second result, and the more interesting one**: fixing the predictor to
actually learn from observed history wasn't a one-line win either. A first
attempt (unbounded cumulative transition counts) produced *identical*
numbers to the naive policy -- traced by hand: two early A-to-B
observations from the steady phase outweighed the first three A-to-A
observations inside the burst, so the predictor never flipped its guess
before the 4-round burst had already ended. Cumulative counting that
weighs ancient history as heavily as the last few rounds isn't actually
adaptive. Fixed by windowing to only the single most recent transition per
tenant (`window=1`) -- this recovers a meaningful chunk of naive's damage
(90.2% vs 87.9%) by correctly predicting "stay put" for the second half of
each burst, once it's seen one same-tenant repeat.

**Third result, the one worth sitting with**: even the working adaptive
predictor doesn't beat reactive-only. Every regime boundary (4 of them in
this schedule) is an unavoidable miss for *any* online predictor -- nothing
can see a burst starting before it starts -- and because the shared budget
here is tight (Phase 8: 750MB, barely enough for tenant B alone), a wrong
prefetch is expensive: it evicts the correct tenant's already-warm,
already-sufficient cache to make room for a guess that turns out wrong,
paying a full tenant re-fetch for nothing. The predictor's *correct*
guesses (mid-burst) save less than its *incorrect* guesses (at every
boundary) cost. Reactive's advantage isn't foresight -- it's never making
a confident bet at all, which turns out to be the right call under this
combination of tight budget and boundary-heavy workload.

This is not a case for abandoning prediction -- it's evidence for a
specific, sharper design than "always prefetch a guess": a real policy
should act only when it's actually confident (e.g., require several
consecutive consistent observations before trusting a transition, not
just one), and the *value* of prefetching should be weighed against the
*cost* of being wrong, which this experiment shows is not small when
cache budget is tight. That confidence-gated version was not built or
tested here -- a reasonable next step, not a conclusion this project's
evidence yet supports one way or the other. What the evidence does
support, plainly: **naive prediction is worse than no prediction, working
prediction is better than naive prediction but still worse than no
prediction, and the deciding factor is the asymmetric cost of being wrong
under a tight shared budget** -- a more precise and more honest finding
than either "ML helps" or "ML doesn't help" would have been.

### Phase 10 follow-up: gating on confidence, not just improving the guess

The natural next question the section above raised: if wrong guesses are
what cost the most, what happens if the predictor only acts when it's
actually confident, instead of always betting on its single best guess?
Added a fourth policy, `ConfidenceGatedPredictor`
(`scripts/run_multi_tenant_dynamic.py`): only prefetch when the last
`confidence_threshold` (2) observed transitions from the current tenant
unanimously agree; otherwise do nothing for that round, falling back to
reactive behavior rather than acting on ambiguous evidence.

| Scheme | Hit rate | Evictions | Prefetches triggered |
|---|---|---|---|
| Reactive | **93.1%** | 62 | 0/16 |
| Confident (threshold=2) | 90.8% | 86 | 6/16 |
| Adaptive (window=1) | 90.2% | 93 | ~14/16 |
| Naive | 87.9% | 115 | 16/16 |

A clean, monotonic result. Confidence-gating is the best of the three
active prefetching strategies, and by a wide margin over naive -- while
acting on only 6 of 16 rounds versus naive's 16/16. This directly confirms
the mechanism proposed above: most of the damage other policies do comes
from acting on weak or stale evidence, not from prefetching itself, and
simply refusing to bet when unsure recovers most of that cost. It still
falls short of reactive -- the boundary-miss problem (no online policy can
foresee a regime change before it happens) isn't something confidence
gating can fix, since the *first* wrong guess after any transition still
looks confident based on everything seen so far.

**The complete Phase 10 ordering**: reactive > confident > adaptive >
naive. Not a verdict for or against prediction in general -- a
demonstration that under a tight shared budget, the *policy around* a
prediction (whether and when to act on it) matters at least as much as
the prediction's raw accuracy, and that getting both right still runs into
a hard limit no amount of cleverness removes: an online policy cannot
avoid paying for the one thing it cannot possibly know, which is that the
world is about to change.
