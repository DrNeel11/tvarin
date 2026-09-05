# Ramer Product Hypotheses

Companion to `docs/ARCHITECTURE.md` (the technical research). This document
does not contain customer discovery findings -- no interviews have
happened yet, that's still the user's own next step. It does now contain
findings from public research (forum threads, tool documentation) done in
this session -- a real but limited and biased source (skews toward
documented/searchable problems and vendor docs, not a representative
sample of users, and involves no back-and-forth to probe further). Treat
it as a first-pass sanity check that narrowed the hypotheses, not as a
substitute for talking to people.

Ranked by the interest levels already agreed on: edge AI highest,
consumer local AI second, datacenter multi-tenant serving explicitly
avoided (Prism/kvcached already occupy it at a scale this project cannot
out-build -- see `docs/ARCHITECTURE.md`'s Competitive Landscape section).

## Findings from Experiment A: measuring Ollama directly (2026-09-03)

The user's own machine already had Ollama installed with 3 pulled models
(qwen2.5:3b-instruct, qwen2.5:7b-instruct, qwen2.5vl:7b) -- real
measurements, not docs, on the same RTX 5060/32GB machine as the rest of
this project's research.

**Ollama already caches something like Phase 5's mechanism -- at the OS
level, not the application level.** Cold-loading qwen2.5:3b-instruct took
24.99s (disk read + decode setup). Loading qwen2.5:7b-instruct next
evicted 3b entirely (only one model fit at a time by default here).
Reloading 3b afterward took **1.62s -- a 15x speedup over its own cold
load**, and this time both models stayed resident together (6.9GB, fits
in ~7GB usable VRAM) rather than evicting each other. The 15x speedup
almost certainly comes from Windows' file-system page cache (the model
file's bytes were already in RAM from the first read), not any explicit
weight cache inside Ollama itself. This is functionally similar to Phase
5's finding but happens for free, uncontrolled, and only holds as long as
system RAM has room to keep the file pages cached -- which narrows
Ramer's potential contribution specifically to scenarios where that stops
holding (combined model sizes exceeding free system RAM, or RAM pressure
from other applications evicting the OS cache).

**Ollama already does automatic CPU/GPU split when a model doesn't fully
fit** -- observed directly: loading a 3rd model while VRAM was nearly full
produced a 13%/87% CPU/GPU split automatically, no configuration needed.

**The most important finding of this pass**: at that light 13% CPU
offload, decode ran at **65.76 tok/s** -- nowhere near Phase 1's 1.1 tok/s
collapse. Forcing a genuinely comparable ~50%/50% split (`num_gpu=14` on
qwen2.5:7b-instruct, its 28 layers) still got **16.43 tok/s** -- about
**15x faster than Phase 1's HF-Transformers-based measurement of the same
kind of offload severity.** This gap is not explained by anything this
project's tiering/caching work addresses -- it's the difference between
llama.cpp (a production C++ inference engine, what Ollama runs) and HF
Transformers' eager Python `generate()` loop (chosen in this project
deliberately for research flexibility, per `docs/ARCHITECTURE.md`'s
framework-choice reasoning, at a real, now-quantified efficiency cost).

**Why this matters more than the other findings so far**: it doesn't just
narrow a hypothesis, it reduces the size of the *entire underlying
problem* this project's numbers implied. The "27 -> 1.1 tok/s" collapse
that Phase 1 used to motivate the whole tiering/caching investigation is,
at least in significant part, an artifact of the framework choice, not a
property of CPU-offloaded inference itself. Any product built on this
research needs to benchmark against llama.cpp's own offload performance as
the real baseline (~16 tok/s at 50% offload, measured here), not the
1.1 tok/s number this project's own research reported -- that's a much
smaller, harder gap to close, and changes how large the opportunity
plausibly is for H1/H2 both.

## Findings from Experiment B: discrete-GPU offload sweep via Ollama (2026-09-03)

Experiment A found that the original "27 -> 1.1 tok/s" catastrophe was mostly
a framework artifact (HF Transformers eager vs llama.cpp), which raised the
real question directly: once a production engine's own offload/caching is
accounted for, is there still a memory-capacity/performance problem worth
building on? Tested on the same RTX 5060 (8151 MiB VRAM, confirmed a true
discrete-GPU architecture, not unified memory) using `qwen2.5:14b-instruct`
(Q4_K_M, 9.0GB on disk, 48 transformer layers) -- a model deliberately picked
too big to fully fit in VRAM, via Ollama's native API sweeping `num_gpu`
(the number of layers placed on GPU, the rest on CPU/system RAM), one clean
model reload per data point:

| layers on GPU | VRAM used | decode tok/s | prefill tok/s |
|---:|---:|---:|---:|
| 0 (all CPU)  | 0.0 GB | 5.11  | 33.2  |
| 8            | 2.12 GB | 6.18  | 82.3  |
| 16           | 3.53 GB | 7.27  | 98.9  |
| 24           | 4.96 GB | 9.09  | 128.9 |
| 32           | 6.40 GB | 12.86 | 170.7 |
| **40**       | **7.83 GB** | **19.64** | **252.5** |
| 48 (all GPU), run 1 | 9.30 GB requested | 6.17  | 89.3 |
| 48 (all GPU), run 2, live-monitored | 9.30 GB requested | 4.40  | -- |
| Ollama's own auto default | 6.20 GB | 12.21 | -- |

**Within its supported range, llama.cpp's own offload is exactly what
Experiment A implied: smooth and monotonic.** 0 -> 40 GPU layers is a clean,
graceful curve in both decode and prefill, no artificial software cliff.
This is the production engine doing its job well -- consistent with, and
reinforcing, Experiment A's finding that Phase 1's original catastrophe was
a framework artifact, not a fundamental property of CPU-offloaded inference.

**But there is a real, sharp, reproducible cliff -- exactly at the physical
VRAM boundary, not before it.** Requesting all 48 layers needs ~9.3GB, more
than the card physically has (8151 MiB / ~8.55GB). Decode collapsed to 6.17
tok/s (run 1) and 4.40 tok/s (a second, live-monitored run) -- a 3.2x-4.5x
drop from the 40-layer optimum, and *worse than running fully on CPU*
(5.11 tok/s) in the second trial. Live `nvidia-smi` sampling during that
second run confirmed the mechanism: GPU memory pinned at 7758 MiB (the
card's real usable ceiling) at 95-100% utilization for the whole run --
consistent with Windows' WDDM driver silently paging the VRAM overflow
through system RAM over PCIe rather than erroring out or refusing the
request. The run-to-run variance itself (6.17 vs 4.40 tok/s for the
identical request) is further evidence of paging/contention, not a clean
deterministic slowdown.

**Ollama's own "safe" default leaves real throughput on the table.** With no
`num_gpu` override, Ollama picked a conservative 6.20GB split -> 12.21 tok/s.
The actual achievable optimum on this exact hardware was 7.83GB -> 19.64
tok/s -- **61% faster**, sitting entirely within physical VRAM, that the
default never finds. Ollama isn't probing for the real ceiling; it appears
to reserve a fixed safety margin and stop there.

**This is a real, measured, reproducible gap that survives after accounting
for the production engine's own offload behavior -- and it is entirely
local.** Per the instruction not to assume network memory is the answer:
network memory never enters the picture here. The actual bottleneck is (a) a
static, conservative placement heuristic that leaves ~40% of achievable
throughput unclaimed, and (b) an unmanaged, invisible cliff exactly at the
VRAM/system-RAM boundary that can make GPU offload *worse than not using the
GPU at all*, silently, with no warning. Both are single-machine placement
problems, not cross-machine memory-tiering problems -- a materially
narrower shape than the project's original "remote memory over a network
tier" framing.

**Honest assessment of scope, not just existence.** The gap is real but
shallow in a specific sense: the fix is a bounded, mechanical
improvement -- empirically probe (once, cached per model+quant+context+
machine) the true usable VRAM ceiling instead of trusting a static estimate,
and refuse to silently oversubscribe past it. That's a plausible single
upstream PR for Ollama/llama.cpp, not a deep architectural moat. It answers
the literal experiment ("is there a meaningful gap after accounting for the
engine's existing behavior") *yes*, but it's a narrow VRAM-boundary
auto-tuner, not the broad memory-orchestration platform the project
started out investigating.

**Decision point.** The rule set going into this experiment was: if B finds
a real gap, stop researching and build the first product around it. B did
find one -- but it's worth being precise about what kind of gap before
treating that rule as satisfied. A VRAM-ceiling auto-tuner is buildable in
an afternoon, would plausibly ship as a small upstream contribution to
Ollama/llama.cpp rather than needing a new company, and doesn't touch any
of this project's actual research contributions (Phases 3-10's tiering,
caching, prefetching, scheduling work) -- it's a different, much smaller
problem that happens to share the word "memory." Recommendation: don't
treat this as the wedge to build a company on by itself. It's real signal
that VRAM-boundary placement is under-tuned across the ecosystem (worth
possibly shipping as a free tool or upstream PR for goodwill/credibility),
but H1/H2/H3 still need actual customer interviews before committing
engineering time to a product -- Experiment B closes out the "kill or
validate via more solo experiments" phase without producing a company-scale
wedge on its own.

**Deferred: a possible future confirming experiment, not run.** Post-close,
verified that Qwen3.8-27B (Qwen's current open-weight release as of
2026-09; Qwen3.7 turned out to be cloud/API-only, no open weights) is a
real, downloadable, much larger model than the 14B used in Experiment B --
a candidate for repeating the offload sweep at a scale where VRAM
undersizing is more severe. Deliberately not run: the underlying mechanism
Experiment B found (driver-level VRAM-boundary paging, a static placement
heuristic) isn't architecture- or size-specific, so a 27B repeat would
almost certainly reproduce the same qualitative shape rather than change
the verdict above. Worth revisiting only if customer interviews surface a
reason a bigger model specifically matters (e.g. someone's actual workload
is a 20B+ model, not a smaller one) -- not as a default next technical
step.

## Ramer Local V0: built, not just hypothesized (2026-09-04)

Per the "you're customer #1" reframe -- built the smallest real version of
Experiment B's finding: `scripts/ramer_local.py`, a CLI that sweeps
`num_gpu` for a given Ollama model on this machine, picks the
best-observed split, and caches it (`probe`, `run`, `show-cache`
subcommands). Two real, previously-unknown things surfaced in the first
hour of actually using it, not just theorizing about it:

1. **Off-by-one in the naive version of this exact idea.** `num_gpu`
   counts transformer blocks *plus* the output/embedding tensor -- so
   `num_gpu = block_count` (what `/api/show`'s `model_info` reports) is
   one layer short of true full offload. The first working version of the
   tool recommended `num_gpu=28` for a 7B model that fits entirely in
   VRAM, measured it as ~20% slower than Ollama's own auto default, and
   would have shipped that as "the optimal split" if the tool hadn't also
   measured auto for comparison. Fixed by sweeping to `block_count + 1`.
   Lesson: always include Ollama's own choice as a literal candidate in
   the comparison, not just a reference number -- a hand-picked
   alternative can look like a win by a measurement bug alone.
2. **The optimal split isn't a fixed property of a model+GPU -- it drifts
   with whatever else is using VRAM right now.** Re-running the identical
   probe in different desktop sessions saw free VRAM at probe time range
   from ~7 GB to ~300 MB (browser tabs, widgets, other apps), and the
   measured peak shifted layer-to-layer accordingly (33-41 one run, a
   cleaner 37 another). A "probe once, cache forever" design can hand back
   a split that was safe when cached and lands inside the cliff later.
   V0 now records free VRAM at probe time and warns (not blocks) at `run`
   if free VRAM has dropped meaningfully since -- a real product
   requirement (session-aware placement, not static placement) discovered
   by using the tool, not by predicting it in advance.

Best observed result matches Experiment B's shape: on `qwen2.5:14b-instruct`,
recommended `num_gpu=37` got ~19 tok/s vs auto's ~12.4 tok/s -- consistent
with, though not identical in absolute numbers to, Experiment B's manual
sweep (num_gpu=40, 19.64 tok/s), because the actual peak location moves
with current VRAM pressure as described above.

## Findings from public research (2026-09-03, not interviews)

Two findings that narrow H1 and H2 further, found by reading real forum
threads and tool documentation rather than assumed:

**H2 is more crowded than the original hypothesis assumed.** Confirmed
directly: Ollama -- the single most common consumer local-LLM tool, the
exact audience H2 targets -- already supports multiple models loaded
simultaneously (`OLLAMA_MAX_LOADED_MODELS`) and, when VRAM is full and a
new model is needed, evicts whichever model has been idle longest (LRU).
That is Phase 8's core mechanism, already shipped in the tool this
hypothesis's target users almost certainly already have installed. H2's
"why existing tools aren't enough" claim needs to narrow to something
Ollama's LRU specifically doesn't do -- candidates: does Ollama's eviction
go to a RAM cache (Phase 5's mechanism, ~10x win over reload) or fully
unload and reload from disk each time (much worse, and unverified either
way here -- check before relying on it); cross-tenant scheduling nuance
from Phase 8's *heterogeneous* working-set-size result; Phase 10's
confidence-gated prefetch refinement. Broad "multi-model memory
management for consumer GPUs" is not an open gap; a specific mechanism
within it might still be.

**H1 has a real architecture mismatch worth resolving before pursuing
it further.** Real Jetson-class edge developer forum threads (NVIDIA
Developer Forums, e.g. a Jetson Orin Nano Super user unable to load models
despite "sufficient RAM," root-caused by NVIDIA staff to memory
*fragmentation* -- "the largest contiguous block is very small" -- not a
capacity shortfall) reveal that the most common, accessible edge AI
hardware (Jetson-class SoCs) uses **unified memory**: CPU and GPU share
one physical RAM pool. This project's entire mechanism (Phases 3-10)
assumes *moving data between distinct physical pools* -- VRAM tier, RAM
tier, remote tier. That doesn't describe a unified-memory device at all;
the problem there is allocator fragmentation in one shared pool, a
different technical problem this project has not built anything for. H1
may still be real for edge devices with a genuinely discrete GPU (an
industrial PC with a separate GPU card, not a Jetson-style SoC) -- but
"edge AI" as a category needs splitting by memory architecture before
assuming Ramer's mechanism applies, not treated as one hardware class.

**What was confirmed, not just narrowed**: the general problem class is
real and people hit it unprompted, without any leading. Both a Jetson
developer ("unable to load large models... despite sufficient RAM") and a
consumer-GPU user (32GB V100, Llama-3.1-70B, "Still OOM Memory!" after
trying suggested fixes) describe genuine, undirected frustration with
models not fitting and workarounds not working well. That's real signal
that the pain exists -- it just doesn't by itself tell you Ramer's
specific mechanism is the fix, which is exactly what the interviews still
need to establish.

**Update (2026-09-05), relayed from a Reddit thread analysis, not
independently verified here.** The user brought back a summary of a
r/LocalLLaMA thread (~300 upvotes, ~452 comments) asking why the
subreddit's visible hardware skews so much higher-VRAM than what ordinary
people actually own. The commentary quoted is consistent with something
well known about that community and worth taking at face value even
unverified: r/LocalLLaMA is a **selection-biased enthusiast bubble** --
commenters there describe it as a "world bubble minority," note "99%
wouldn't know what VRAM is," and call the visible high-end hardware
"survivorship bias" / "the crazy end of the pool." Underneath that bubble,
plenty of commenters described running local AI on genuinely constrained
8-16GB VRAM, using the workarounds you'd expect (quantization, CPU
offload, MoE, multi-GPU) rather than upgrading -- and one described
running local models on modest hardware specifically to build experience,
while still paying for cloud LLM subscriptions alongside it (a **local
when possible, cloud when necessary** hybrid workflow, not a
local-replaces-cloud one).

This sharpens the target user further than H1/H2 already had: not
"everyone who wants to run AI locally" (too broad, and most people
outside the enthusiast bubble aren't reachable or motivated by this at
all), but **someone who already runs local AI, is capacity-constrained
(8-16GB-class VRAM), and is bothered that their GPU isn't used well** --
which is, concretely, the user themselves, reinforcing why using Ramer
Local V0.1 firsthand is still the right next validation step rather than
another scrape. Explicitly two separate claims, only one of which this
narrows further: **"VRAM is scarce for a real population of local-AI
users" stays validated** (consistent with everything found earlier in
this section); **"people would pay for a VRAM-optimization layer" is
still unvalidated** -- nothing in this thread speaks to willingness to
pay, only to the workaround-tolerance of people who already run local AI
without one. That distinction is the reason the next input should be
`show-log` data from real usage, not more forum reading.

---

## H1: RAM-constrained edge/embedded AI (highest priority)

**Who has the problem**: engineers building AI into physical products with
fixed hardware budgets set at design time -- robotics, industrial IoT,
medical devices, autonomous vehicles, on-prem security/vision systems.
Unlike a developer's PC, the hardware can't be upgraded post-deployment;
whatever memory ships in the unit is permanent.

**What exactly hurts**: they need local inference (latency, privacy,
unreliable/no connectivity, bandwidth cost rule out cloud), but the
model that would give acceptable accuracy/capability doesn't fit the
unit's fixed VRAM+RAM budget.

**Current workaround** (hypothesized, not confirmed): ship a smaller/more
aggressively quantized model than they'd prefer; over-spec the hardware BOM
at design time (real cost per unit, multiplied across a production run);
in some cases, a cloud fallback despite the reasons they wanted local.

**Why existing tools aren't enough**: Prism/kvcached and the
datacenter-serving ecosystem assume cluster infrastructure and an ops team
-- not a single embedded device with a fixed, often non-GPU or
small-GPU compute budget. Nothing found in the competitive research
targets "make one resource-constrained physical device run a bigger local
model."

**What Ramer would do**: transparent VRAM/RAM(/local flash) tiering tuned
for this project's actual finding -- Phase 5's persistent caching works
well specifically when the overflow fits in combined VRAM+RAM; Phase 1
already measured where that stops being true on one reference machine.
The pitch is narrow and specific: not "infinite memory," but "get
meaningfully more usable model capability out of a fixed embedded budget
than naive offload gives you," with the working-set cliff (Phase 5) stated
as an honest limit, not hidden.

**Experiment that could prove it**: benchmark a realistic edge-class
device (e.g. a Jetson-class or similarly constrained board, not just this
project's desktop GPU) running a model sized to overflow VRAM, comparing
naive CPU offload against Ramer's tiering+caching, on a task representative
of the target use case (not just token throughput -- latency percentiles
likely matter more here than aggregate tok/s).

**What would falsify it**: if edge AI teams report they simply always ship
the smaller model that fits and have no interest in a larger one at
somewhat reduced speed; if their actual bottleneck is power draw or
compute (not memory); if the hardware BOM cost of "just add more RAM" at
design time is cheap enough that this is a non-problem in practice.

---

## H2: Consumer/hobbyist local AI on 8-12GB GPUs

**Who has the problem**: individual developers, students, and indie
builders running local models on consumer GPUs (RTX 3060/4060/5060-class,
8-12GB) -- this project's original framing.

**What exactly hurts**: can't run larger models (14B+) at usable speed
(Phase 1 measured this directly: on this project's reference 8GB/32GB
machine, usable local inference stops somewhere between 7B and 14B); can't
run an agent/RAG stack (a chat model plus an embedding model plus a
reranker, say) simultaneously without exceeding VRAM.

**Current workaround**: use hosted APIs instead of local for anything
bigger; run only quantized/smaller models; buy a bigger GPU; rent cloud
GPU time (RunPod and similar).

**Why existing tools aren't enough**: Ollama/llama.cpp already solve
reasonable single-model CPU/GPU splitting well (this project's Phase 1
essentially re-derived why that ceiling exists, not that it's unsolved).
What they don't do is Phase 8's multi-model coexistence -- and Prism/
kvcached, which does, isn't packaged for a single personal machine or as
simply as `ollama run`.

**What Ramer would do**: a `ramer run <model>` CLI that applies Phase
3-5's tiering/caching automatically, with Phase 8's shared-cache mechanism
extended to let a hobbyist run 2-3 models concurrently (a chat model + an
embedder, say) on one consumer GPU without manually managing which is
loaded.

**Experiment that could prove it**: same technical shape as H1 but on
consumer hardware -- benchmark running 2-3 concurrent local models on one
8GB GPU via Ramer against the naive alternative (sequential loading /
manual juggling) on the same hardware.

**What would falsify it**: if hobbyists report they're satisfied running
one model at a time, or default to cloud APIs for anything demanding
enough to need this; if Ollama/LM Studio already handle the common case
well enough that nobody notices the gap this targets.

---

## H3: Small on-prem/compliance-constrained AI teams

**Who has the problem**: small teams (legal, healthcare, finance,
government-adjacent) who can't send data to cloud LLM APIs for compliance
reasons, running several specialized local models on a small number of
GPUs (1-4) -- bigger than a hobbyist's one machine, much smaller than the
GPU clusters Prism/kvcached assume.

**What exactly hurts**: need multiple specialized models (e.g. a coding
assistant, a document Q&A model, an embedding model) on a constrained GPU
budget, without the engineering headcount to operate cluster-scale serving
infrastructure built for ops teams.

**Current workaround** (hypothesized): sequential model loading via manual
scripts; over-provisioning GPUs beyond what utilization would otherwise
justify, because the alternative (engineering a scheduler) isn't worth
the team's time at this scale.

**Why existing tools aren't enough**: Prism/kvcached's own framing
("organizations," cluster GPU utilization, ops-team-oriented) suggests a
real adoption barrier at this smaller scale -- worth confirming directly,
since it's the least validated of the three hypotheses here.

**What Ramer would do**: the same Phase 8 mechanism as H2, framed and
packaged for a small IT/ML team rather than either a single hobbyist or a
cluster ops team -- the operative bet is that there's a scale in between
where existing tools are simultaneously overkill (Prism) and absent
(nothing packages this for "a few GPUs, a few specialized models").

**Experiment that could prove it**: a deployment-shaped demo (not just a
benchmark) -- install-and-run simplicity comparable to H2's CLI, but
measured for a multi-specialized-model workload on 2-4 GPUs.

**What would falsify it**: if such teams say they'd rather just buy more
GPUs than adopt new infrastructure (likely, if the team is small and GPU
budget isn't the binding constraint); if compliance requirements turn out
to make them risk-averse toward adopting new/unproven memory-management
software specifically (a real possibility worth asking about directly).

---

## Explicitly not pursuing (for now)

**Datacenter multi-tenant LLM serving.** Phase 8's core mechanism is
already solved there by Prism/kvcached, in production at 10,000+ GPU
scale, published at OSDI'26. Revisit only if H1-H3 all fail *and* a
specific gap within that space is found that Prism doesn't cover -- not
as a default fallback.

## How to use this document

**Research phase is complete as of Experiment B (2026-09-03), tagged
`v0.1-research-complete` -- see `docs/ARCHITECTURE.md`'s "Status" section.**
No further solo technical experiments are planned. The exhausted technical
hypotheses (remote memory, layer tiering, persistent caching, working-set
limits, batching, multi-tenant sharing, tenant-level prefetch, changing
workloads, prediction, cross-architecture generality, discrete-GPU
behavior, physical VRAM limits) have narrowed the space enough that another
solo experiment is worth less right now than real customer signal.

**Next milestone: ~20 customer-discovery interviews**, then identify the
top 2-3 recurring, *unprompted* pain points before deciding what (if
anything) to build. Don't pitch Ramer or ask "would you use an AI memory
orchestration system?" -- that produces polite, useless data. Ask about
their actual workflow instead:

1. What hardware are you running?
2. What models do you actually use?
3. What's the largest model you've tried to run locally?
4. What happened when it didn't fit?
5. What did you do instead?
6. How long did that take / how much did it cost?
7. Do you run multiple models at the same time?
8. Do you use Ollama, llama.cpp, vLLM, LM Studio, etc.?
9. What's the most frustrating part of running AI locally?
10. If you could magically fix one thing about your setup, what would it be?

Then shut up and listen. A hypothesis is confirmed by an *unprompted* pain
description matching it (H1, H2, H3 above, or something not yet
considered), not by someone agreeing when asked directly if the problem
sounds bad. Watch specifically for *why* behind any "I run out of VRAM"
answer -- it could resolve to model weights (this project's framing), KV
cache growth, running multiple models, image/video models, or agentic
multi-model/tool workflows, each implying a different, more specific
product than "memory orchestration" in general.

For each pain point that recurs, work it through before building anything:
who has it -> how frequently -> current workaround -> why that workaround
falls short -> what it costs them -> can a one-week prototype address it ->
would they actually use or pay for that prototype. Only after that chain
holds up for a specific problem does a build decision get made. If the
interviews reveal nothing worth building, that is not a failed outcome --
it means this validation phase did exactly what it was for: preventing
months of work on something nobody needs. Update this file with findings
as they come in.
