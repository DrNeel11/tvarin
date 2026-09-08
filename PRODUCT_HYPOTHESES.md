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

**Boundary case (2026-09-05): the "does Ramer make an impractical model
practical" test, on `qwen2.5:32b-instruct` (19GB Q4_K_M, 64 layers) on the
same 8GB card.** Answer: **no.** Best achievable split (`num_gpu=22`,
7.48GB) got 3.29-3.78 tok/s, only **+6.7%** over Ollama's own auto default
(3.08-3.17 tok/s) -- nowhere near the 14B case's +61%. A one-line prompt
("What is the capital of France?") took over two minutes wall-clock,
almost entirely model-load time, at any split. There just isn't headroom
to unlock here: this model is ~2.3x the 14B one tested in Experiment B,
and every split tested tops out in the 2.4-3.8 tok/s range -- unusable for
interactive use regardless of placement. **Ramer can optimize around a
memory constraint; it cannot repeal the hardware constraint.** Keep this
as the boundary case against overclaiming: the win is real (Experiment
B's 14B case, this V0's own +12-61% results depending on model/session)
but bounded -- it recovers slack Ollama's own heuristic leaves on the
table, it does not turn "doesn't fit" into "fits."

This run also found a real robustness bug in the tool itself: pushing
`num_gpu` far enough into VRAM oversubscription (`num_gpu=54` of 65, here)
didn't just degrade throughput, it crashed that one Ollama request outright
(`HTTP 500`) -- the Ollama *server* survived, but the probe script did not
without a fix. `scripts/ramer_local.py`'s sweep now treats a crashed
candidate as a failed data point (never selected, sweep stops early
rather than dying) instead of letting one bad candidate kill the whole
probe -- caught by actually running the tool against a large enough model,
exactly the kind of finding a bigger real-usage pass was expected to
surface.

## First real-usage sample (2026-09-05, compressed into one session)

Per the 7-day real-usage plan, generated a first small sample now rather
than waiting: 7 real `run` invocations across 14B/32B/7B with varied
prompts (code, explanation, casual, long-form), switching models between
calls to specifically exercise the model-switching and VRAM-drift cases
the plan called out. Raw log: `ramer_local_usage.jsonl` /
`show-log`. Gains ranged from **-21.7% to +77.6%** on the *same* model and
split, depending on what else had recently touched the GPU -- the average
is a real positive, but any single call is not guaranteed to win.

Two honest findings, not smoothed over:

1. **The staleness safety net has a real gap.** One 14B call (a long,
   ~400-token response) landed at 10.0 tok/s -- *worse* than Ollama's own
   auto placement (12.77 tok/s at probe time) -- and the 20-token quick
   check before it did NOT catch this; it only triggered the fallback on
   the *next* call, after more drift had accumulated. A short probe is a
   leading indicator, not a guarantee, especially once a real generation
   runs long enough for throughput to drift mid-response. Noted, not
   fixed yet -- consistent with the "don't immediately fix what you
   notice, write it down" discipline for this usage phase.
2. **The safety net did also work as designed, once.** Immediately after
   the above, switching back to 14B (right after a 32B load) triggered a
   correct fallback: quick check measured 11.5 tok/s vs. the cached 16.9
   tok/s, fell back to auto, ran at 12.18 tok/s instead of risking another
   regression. This is the mechanism from the "V0.1" section working
   exactly as intended, live, not in a designed test.

**Parked idea, deliberately not built (2026-09-05): Ramer as a
placement-decision controller, not a `num_gpu` sweeper.** The false
negative above prompted a reframing worth recording before setting it
aside. The unifying principle: **don't optimize for prediction accuracy,
optimize for the decision** -- the same structure Phase 10 already found
for prefetching (confidence + cost + opportunity-to-hide determine whether
to *act* on a prediction, not just how accurate it is) reappears here
(calibration confidence + reload cost + remaining generation length
determine whether to *change* placement, not just how good the placement
looks). Two honest constraints that keep this grounded rather than
grandiose: (1) `num_gpu` can't change mid-generation without a full
reload, so "continuous adaptation" really only ever means "a good cheap
decision before each call" -- V0.1's quick-check is already a primitive
version of this, not a first step toward it; (2) a cheap improvement to
try first, before anything resembling a learned decision model, is
watching the real per-token rate as an actual response streams out
(no extra calibration cost, since the request is already happening) and
requiring *sustained* degradation (not single-sample noise) before
considering intervention -- and even then, weighing expected benefit
against the cost of an interruption/reload, since finishing a
slightly-slower response can beat interrupting it. **Not building this.**
The trigger to revisit: if the week of real usage shows model-switching /
memory-pressure / long-generation drift repeatedly making the cached
placement unreliable (not a one-off, as seen once so far), that's a
recurring problem with the system's own measurements of the failure
already in hand -- a much stronger starting point than a clever idea
looking for a problem.

**Update, same day: the pattern recurred immediately and was root-caused,
not just observed.** Continued generating usage (switching between
3B/7B/14B/32B repeatedly) and the exact failure above reproduced **4/4
times in a row** -- every single switch back to `qwen2.5:14b-instruct`
after another model triggered the staleness fallback (10.0-11.8 tok/s vs.
16.9 tok/s at probe time). That's not the "wait a week and see if it
recurs" signal the parked idea called for -- it recurred on demand, which
pointed at something mechanical rather than diffuse "environment
changes." Root cause found and confirmed: **Ollama's default `keep_alive`
(~5 min) keeps a previous model resident in VRAM after you've moved on to
a different one**, and nothing in `ramer_local.py` ever unloaded it --
`unload()` only ever targeted the model about to run. Every switch was
running the new model's cached placement while the old one was still
warm and competing for the same VRAM.

This is narrower and cheaper than the parked "placement-decision
controller" idea -- no streaming, no learned model, no sustained-drift
detection needed. Fix: `unload_others()` checks `/api/ps` and evicts
anything else resident before probing or running (`scripts/ramer_local.py`,
commit `ecdf7dc`). Verified fixed **2/2** immediately after: 15.34 and
14.32 tok/s on the identical switch-back scenario that failed 4/4 times
before, no staleness fallback triggered, both real wins over auto
(+20.1%, +12.1%). The parked controller idea should be revisited only if
staleness recurs *after* this fix -- this specific mechanism is now
closed, but it doesn't rule out other causes of drift (true VRAM pressure
from non-Ollama apps, which the fix does nothing about).

**Tested that exact remaining question directly (2026-09-06) -- genuine
external (non-Ollama) VRAM pressure, simulated with a small PyTorch script
holding 2.5-3GB of real VRAM that `unload_others()` cannot see or clear.**
Two things came out of this, one much bigger than intended:

**A major, unrelated latency bug, found by accident.** A simple "what is
2+2" round-trip was taking 22-25 seconds wall-clock regardless of whether
the artificial pressure was active -- the pressure wasn't the cause.
Instrumented every internal call and found each one taking ~2 seconds
longer via this tool than the identical request made with `curl`.
Root cause, confirmed directly and in isolation from Ollama entirely:
`OLLAMA_HOST` was `"http://localhost:11434"`, and Python's `urllib` on
this machine stalls ~2.05s resolving/connecting to `"localhost"` (tries
IPv6, falls back) versus ~0.002s for the identical request to
`"127.0.0.1"` -- a >1000x difference on every single API call. Since one
`run` invocation makes 5-6 calls, this was adding ~10-12s of pure dead
time to *every* call this tool has ever made, unrelated to the model, the
GPU, or VRAM pressure at all. One-line fix (`scripts/ramer_local.py`,
commit `42dc064`): verified the identical call dropped from 22.2s to
8.19s wall-clock. This is a bigger, more universal win than anything else
found today -- it affects every invocation, not just the VRAM-drift
cases -- and it's exactly the kind of thing a designed benchmark would
never surface, because nobody benchmarks wall-clock time on a tool whose
own metric is tok/s.

**The actual question, once the noise was removed: yes, the safety net
generalizes to real external pressure.** With the latency fix in place,
running 14B while a separate process held 3GB of real VRAM triggered the
quick-check correctly (11.79 tok/s measured vs. 16.94 tok/s cached, below
the regression threshold), fell back to Ollama's auto placement, and
completed safely at 13.83 tok/s in 11.5s total. The mechanism generalizes
because it's cause-agnostic -- it measures actual throughput, not "is
another Ollama model loaded," so it doesn't need to know *why* something
degraded to react to it. What it can't do is recover the *optimal*
placement for the new constrained state (auto is safe, not necessarily
best under 3GB less headroom) -- that would need a fresh probe under the
current conditions, which nothing does automatically, and still isn't
being built, per the same discipline as the parked controller idea.

## MoE expert placement: a bigger lever Ramer structurally cannot reach (2026-09-06)

A r/LocalLLaMA thread ("What can you realistically do with 8GB VRAM in
2026?") surfaced something more significant than general sentiment:
multiple commenters described running MoE (Mixture-of-Experts) models --
Gemma-4-26B-A4B, Qwen3.6-35B-A3B -- at genuinely usable speeds on 6-8GB
cards by manually keeping attention/shared layers on GPU and routing
routed-expert FFN weights to CPU, via **llama.cpp or LM Studio directly,
not Ollama**. That's a structurally different lever than anything
`ramer_local.py` does: it sweeps a single `num_gpu` (whole transformer
*blocks*), which cannot distinguish "this tensor is touched by every
token" (attention) from "this tensor is touched by ~1 in 5 tokens"
(a routed expert) within the same block.

Checked whether Ollama exposes this rather than assuming: **it doesn't.**
Three independent confirmations: (1) Ollama's documented `/api/generate`
options list has no expert-offload equivalent -- only the standard
sampling params plus `num_gpu`/`num_ctx`/`num_batch`; (2) Ollama's own
issue tracker has this as an **open, unimplemented feature request**
([#11005](https://github.com/ollama/ollama/issues/11005), dynamic
expert loading, claimed 50-90% potential VRAM reduction, no maintainer
commitment); (3) **direct empirical test** -- pulled a real small MoE
model (`granite3.1-moe:3b`, IBM Granite, 32 blocks, 40 experts/block, 8
active) and swept `num_gpu` with `ramer_local.py probe`. VRAM scaled in
perfectly smooth, linear 0.3-0.4GB steps per layer (0 -> 0.538 -> 0.872 ->
1.215 -> 1.623 -> 2.032 -> 2.366 GB) -- identical in shape to every dense
Qwen2.5 sweep earlier in this project. No discontinuity, no
expert-awareness: Ollama moves whole blocks (all 40 experts included)
per `num_gpu` step, exactly like a dense model.

The underlying engine llama.cpp genuinely has the finer control the
Reddit commenters were using -- `--n-cpu-moe N` (moves routed-expert FFN
weights of the first N layers to CPU, keeps attention on GPU) and
`--override-tensor` (regex-based per-tensor placement) are real, current,
documented flags. **Ollama does not surface either through its API.**
The Reddit commenters achieving the good results were bypassing Ollama
specifically to get this.

**What this means for Ramer, stated plainly**: the MoE lever is real and
probably larger than the dense-model VRAM-ceiling gap Experiment B/V0
already exploit -- but it is not reachable by tuning `ramer_local.py`
harder. `ramer_local.py` is built entirely on Ollama's REST API
(`/api/generate`, `/api/show`, `/api/ps`); closing this gap means driving
a *different* engine process (raw `llama-server`/`llama-cli`) with a
*different* flag Ollama doesn't expose -- not "sweep one more parameter,"
a second backend integration. That is a materially bigger undertaking
than anything built this session, and a real fork in what "Ramer"
structurally is: an Ollama placement optimizer, or a
get-the-most-out-of-your-GPU tool that happens to currently only speak to
Ollama. Not started without an explicit decision to take that on.

**Resolved (2026-09-06): named as the long-term vision, deliberately not
started now.** The precise framing to carry forward: Ramer's real
long-term shape isn't "the tool that finds the best `num_gpu`" -- it's
**an automatic local-inference optimizer that decides which parts of a
model belong in VRAM vs. RAM vs. other tiers, aware that different
architectures (dense vs. MoE, and whatever comes after) need different
placement strategies to get the most usable performance out of the
hardware someone actually has.** `num_gpu` sweeping is this session's
first, narrowest instance of that idea, not the whole of it -- MoE-aware
placement would be a second, structurally different instance (a second
backend, not a parameter). Explicit decision: **the immediate product
stays narrow** (Ollama-API-driven, dense-model placement) until real
customer demand validates the broader problem is worth solving --
matching the discipline already established for the parked
placement-controller idea. This vision statement is for later reference,
not a roadmap being executed now.

## Real agentic-workload validation: mini_agent.py (2026-09-06)

Per the "test Ramer under a real coding-agent workload, not a synthetic
benchmark" idea -- rather than installing a third-party agent (Cline),
built a minimal one (`scripts/mini_agent.py`) using Ollama's native
tool-calling (`/api/chat` + `tools`, the same mechanism Cline/Continue
use, confirmed real since Qwen2.5-instruct models report `tools` in
their capabilities). Sandboxed file read/write/list and shell-command
tools, a real multi-step loop, reuses `ramer_local.py`'s cached optimal
`num_gpu` automatically.

**Real result, not staged**: asked it to write a `divide()` function with
a deliberate zero-division bug plus a test that triggers it. It wrote
both files, ran the test, watched it crash with a real
`ZeroDivisionError` traceback, diagnosed and fixed the bug, and
confirmed the fix by re-running -- the actual "sees errors, fixes them"
loop, not a scripted demo, entirely on Ramer's optimized placement
(`num_gpu=37` for `qwen2.5:14b-instruct`, the same split Experiment
B/V0 already validated).

**A real gap this surfaced, noted honestly**: `mini_agent.py` reads
Ramer's cached `num_gpu` once at startup and uses it for every call in
the session -- it does **not** go through `ramer_local.py`'s
staleness quick-check or streaming drift-watch (`run_with_drift_watch`).
A long agent session (many tool-call round-trips, potentially minutes)
has exactly the kind of extended, evolving-context workload where those
safety nets matter most, and currently gets none of them. Noted, not
fixed yet, consistent with the "observe before building" discipline --
worth doing if agentic workloads become the primary use case, not
before.

## The placement gap under real agentic (tool-calling) load -- corrected (2026-09-06)

**First pass (superseded below, kept for the record).** Ran the identical
non-trivial multi-file task (build a CLI todo app + tests, run them, fix
a real failure until clean) through `mini_agent.py` twice on
`qwen2.5:14b-instruct` -- once forced to `num_gpu=37`, once `num_gpu=auto`
-- to see whether the 61% single-shot gap survives a real tool-calling
session. Both trials landed at ~10.5-11 tok/s per call, within noise of
each other, and were written up as "the gap mostly vanishes under
agentic load," with a hypothesis that `/api/chat` itself was inherently
slower than `/api/generate`.

**That conclusion was wrong, and the investigation that found the error
is the more important result.** Asked directly to "find and help make
this better," went back and tested each suspected mechanism *properly*
-- with repeated trials and controlled, matched inputs, not single
one-off comparisons (the exact mistake this project has caught itself
making before, e.g. the original 20-token quick-check calibration
issue). Findings, in the order tested:

1. **`/api/chat` vs `/api/generate`, repeated 3x each, identical prompt,
   fresh-unloaded before every call: 15.7-15.9 tok/s both ways.** No real
   difference. The original "12.8 vs 19.6" comparison was two single
   samples taken under different, unmeasured conditions -- ordinary
   session-to-session variance (already documented extensively this
   project), not an endpoint effect.
2. **Tools present but not called, repeated 3x: 15.4-15.5 tok/s.** A
   trivial ~2-3% dip from the larger serialized tools schema in the
   prompt, not the mechanism.
3. **An actual tool call generated, repeated 3x: 15.9-16.1 tok/s.** No
   slowdown at all -- if anything slightly faster (noise).
4. **A long (~970-token) prior context, single flat message, repeated
   3x: 14.35-14.47 tok/s.** A real, modest, reproducible ~10% decline
   from context size alone -- attention cost during decode genuinely
   scales with how much is already cached. Confirmed again independently
   by re-running the original agent task a second time: 14.24 tok/s at a
   541-token prompt declined smoothly to 10.5 tok/s at 4062 tokens,
   right at the 4096 default `num_ctx` ceiling.
5. **The decisive test: a controlled 2 (context size) x 2 (placement) x
   2 (repeat) grid, using fixed synthetic context (not model-driven, so
   both conditions see byte-identical input) instead of letting the
   model's own unpredictable conversation shape the comparison:**

   | context | num_gpu=37 | num_gpu=auto | gap |
   |---|---:|---:|---:|
   | small (229 tok) | 15.12, 15.24 | 11.91, 11.73 | **+28%** |
   | large (2115 tok) | 13.21, 13.17 | 9.57, 9.64 | **+37%** |

   Clean, non-overlapping, reproduced twice at each cell. **The
   placement gap is real, substantial, and does not shrink with
   context -- it holds at roughly 30-40% at both sizes tested,** while
   context growth (item 4 above) slows *both* placements down by a
   similar relative amount without closing the gap between them.

**Corrected conclusion, replacing the "mostly vanishes" claim above**:
the original `mini_agent.py` A/B (3 calls each, model-driven trajectory)
was underpowered and likely confounded by ordinary session variance or
something specific to that one real run (actual file I/O and subprocess
execution happening between LLM calls, unlike the pure-chat controlled
tests) -- not evidence that chat/tool-calling itself defeats the
placement win. The controlled, repeated, matched-input testing says the
opposite: **Ramer's placement optimization does transfer to real
chat/tool-calling workloads, gap intact.** What's actually confirmed to
matter independently is context-length-driven decode slowdown (a
real, separate, smaller effect, affecting all placements alike) -- worth
keeping in mind for future work on context/KV-cache-aware placement, per
the long-term vision already recorded above, but it is not the
"agentic workloads defeat Ramer" story the first pass reported. That
first pass is left in this document rather than deleted, because getting
caught, corrected, and rewritten in the open is more useful here than a
clean-looking record.

## Context-aware placement: from parked idea to real failure, fix, and validation (2026-09-06)

The "optimal allocation isn't just weights-on-GPU, it's weights *and*
KV cache for the actual workload" thesis recorded under the long-term
vision stopped being hypothetical the first time `mini_agent.py` was
used on a real project. Chain of events, all in one sitting:

1. **Real use surfaced a real UX bug.** Pointed at `D:\fl_ai` (a real
   FL Studio mixing-agent codebase), the agent read two source files
   then went silent for minutes while generating a full rewritten file at
   ~10 tok/s -- indistinguishable from a hang, and a 300s urllib timeout
   was killing exactly this kind of healthy long generation (an earlier
   "timeout" wrongly blamed on GPU contention was almost certainly this).
   Fixed by streaming `/api/chat` (text appears live, a token counter
   ticks during tool-call generation, and the timeout now bounds gaps
   *between* tokens, not the whole response).
2. **Reading real files needs more context than Ollama's 4096 default**
   -- past it, Ollama silently drops the *oldest* messages, i.e. the
   system prompt and the task itself. Added `--num-ctx` (default 8192 for
   the agent) plus a warning at 85% of the window.
3. **The larger context immediately broke the placement.** First run at
   8192 with the cached `num_gpu=37` (probed at 4096): call 1 at 8.71
   tok/s, call 2 at **1.47 tok/s** -- deep cliff. Diagnosis confirmed by
   measurement, not inference: at 4096 that split occupies 7.27GB; at
   8192 it occupies **7.89GB** (`/api/ps`), nvidia-smi at 7787/8151 MiB --
   the +0.62GB is the doubled KV cache, and it lands the split past the
   ~7.76GB usable ceiling Experiment B identified. The VRAM math from
   the original research predicted this exactly.
4. **Fix: a placement is only valid at the context it was probed at.**
   `ramer_local.py` now takes `--num-ctx` for both `probe` and `run`,
   passes it into every measurement, and includes it in the cache key --
   so a split measured under one memory pressure can never be silently
   reused under another. `mini_agent.py` looks up the placement for *its*
   `num_ctx` and falls back to Ollama's auto (which does account for KV
   cache) with a hint when none exists. Older cache entries simply stop
   matching and get re-probed once.
5. **The context-aware probe found the right answer -- then exposed a
   flaw in how the winner was picked.** At 8192 the sweep correctly found
   fewer layers fit: `num_gpu=33` at 7.116GB, +23.5% over auto. But the
   first run's raw argmax chose `num_gpu=41` at a *reported 8.68GB* --
   past the card's physical VRAM, already being paged -- because it
   measured 0.04 tok/s faster than 33. Noise buying a zero-headroom
   placement, the exact configuration every drift/cliff failure this
   project has seen comes from. Added a headroom-aware tiebreak
   (`pick_best`): among candidates within 3% of the fastest, take the
   smallest VRAM footprint. Verified offline against the identical grid
   data (now selects 33), then by a real `--force` re-probe.
6. **Before/after on the identical task at 8192**: stale placement 8.71 /
   **1.47** tok/s, 85.9s total; context-matched placement 12.51 / **12.09**
   tok/s, 24.1s total. 3.6x overall, 8x on the call that had cratered.

**What this settles**: context-aware placement is not a future feature,
it is a correctness requirement -- a placement tuned without regard to
the context window is *wrong*, not merely suboptimal, and the failure
mode is a 5-8x throughput collapse that shows up the moment a real
workload (an agent reading real files) needs a real context window. The
gain over auto is smaller at 8192 (+23.5%) than at 4096 (+61%) because
the KV cache leaves less slack to reclaim -- honest, and consistent with
the original thesis that the budget is shared. The tiebreak rule
generalizes beyond this case: with measurement noise of a few percent
and a cliff this sharp, preferring headroom on ties is the right default
everywhere, not just at large contexts.

## Experiment C: context-length sweep at fixed model/hardware (2026-09-06)

Direct follow-up to the context-aware placement fix above, proposed by the
user as the next experiment: same model (`qwen2.5:14b-instruct`), same
GPU, sweep only `num_ctx` in `[2048, 4096, 6144, 8192, 12288]`. Two series
per context, both via `ramer_local.py`'s own `probe()`/`measure()`:

1. **Optimized** -- a full fresh probe at each context (what Ramer would
   actually recommend there).
2. **Fixed** -- `num_gpu=37` (the split chosen at the smallest context,
   2048) held constant and re-measured at every larger context, to isolate
   what changes when only the KV cache grows. VRAM in this series moves
   only because of KV cache (weights are pinned), so its growth relative
   to the ctx=2048 baseline is a usable proxy for KV-cache size -- Ollama's
   API doesn't expose that number directly.

| ctx | auto tok/s | best num_gpu | best tok/s | gain vs auto | fixed(37) tok/s | inferred KV growth |
|---|---|---|---|---|---|---|
| 2048 | 12.70 | 37 | 16.64 | +31.0% | 15.22 | +0.0 GB |
| 4096 | 12.96 | 37 | 16.55 | +27.7% | 16.12 | +0.29 GB |
| 6144 | 12.07 | 33 | 14.02 | +16.2% | 14.42 | +0.60 GB |
| 8192 | 11.64 | 33 | 13.98 | +20.2% | 16.46 | +0.91 GB |
| 12288 | 10.11 | 41 | 13.70 | +35.5% | 13.93 | +1.54 GB |

Full grids and raw JSON: `results/expC_context_sweep.jsonl`,
`results/expC_context_sweep_summary.json`.

**Finding 1 -- the gain over auto is robust across context length, not a
4096-specific fluke.** Every context in the sweep beat auto by
16-35%, with no trend toward the gain vanishing as context grows (12288
is actually the largest gain in this sample). Ollama's auto placement
gets *slower* as context grows (12.70 -> 10.11 tok/s, auto probably
holding back layers to leave more room for the larger KV cache) while
Ramer's chosen placement stays in a tight 13.7-16.6 tok/s band across the
whole range -- the win holds because Ramer keeps re-optimizing for the
actual budget instead of applying one static conservative margin.

**Finding 2 -- KV-cache growth for this model was smooth, not a cliff by
itself.** +1.54GB from 2048 to 12288 (6x the context) is gradual, in line
with the VRAM math from Experiment B, and would never by itself have
been surprising.

**Finding 3 -- and yet the fixed-37 series never collapsed, even past the
card's physical VRAM.** At ctx=12288 the fixed placement reported
8.521GB of VRAM in use -- *above* this 8151MiB card's total capacity, the
same kind of overflow that produced the 1.47 tok/s catastrophic collapse
documented in the section above (same model, same num_gpu=37, at a
smaller 8192 context that only reached 7.89GB). Here it just... didn't
collapse: 13.93 tok/s, close to the ctx=2048 baseline. **This is the
important negative result of this experiment**: crossing the physical
VRAM ceiling is necessary but not sufficient to trigger the cliff --
something else (external GPU load at the time, driver page-out timing,
which pages get evicted) determines whether a given overflow is smooth
degradation or a 5-10x collapse, and that something isn't captured by
context length, VRAM headroom, or any other static number probed here.

**What this settles and what it doesn't**: context-aware caching (keying
placements by `num_ctx`) is still clearly correct -- it removes a
placement that's flatly wrong for the current context, which is real and
common. But this experiment shows it can't be sold as "prevents the
cliff," because the cliff isn't a deterministic function of the inputs
Ramer can see ahead of time. That's not a new finding so much as this
project's second confirmation of it (the first was the earlier
non-deterministic-collapse note on the parked placement-controller idea)
-- but this time from a clean, designed sweep rather than an accidental
observation. It reinforces that the *runtime* safety nets (`run`'s
20-token quick-check, and `run_with_drift_watch`'s mid-generation
detection) are doing real, load-bearing work and are not a redundant
belt-and-suspenders on top of context-aware caching -- they're the only
thing in this system that can actually catch the nondeterministic case,
because nothing measured in advance predicts it reliably.

## mini_agent.py gutted real, working code on a real project (2026-09-08)

Continued the "use it yourself, on real work" plan from a fresh angle: a
genuinely useful, bounded, real task on `D:\fl_ai` (an unrelated real
project) -- wire an already-collected-but-never-read `chat_log` into the
system prompt so `/api/chat` gets real multi-turn conversational memory,
a real gap confirmed by grep before starting (`chat_log.append(...)` in
`server/main.py`, never referenced anywhere else in the codebase).

**Run 1 stalled, not just slowed.** ~250s and 866 tokens into generating
a plan, the socket went fully silent and the 300s per-read timeout
(added specifically so streaming distinguishes a healthy slow generation
from a real hang) fired -- a genuine stall, not the "healthy long
generation" case that fix targeted. Ollama was confirmed idle and
healthy immediately after (`/api/ps`, `/api/tags`, `nvidia-smi` all
normal, correct 7.1GB VRAM resident). One data point, not chased further
-- but `mini_agent.py` crashed with a raw traceback instead of a clean
message, so that got fixed regardless (`chat()`'s `TimeoutError`/`OSError`
now stop the session cleanly with a diagnostic and a logged `stalled:
true` entry, same "record the failure, don't crash" discipline as
`ramer_local.py`'s `safe_measure`).

**Run 2 didn't stall -- it did something worse.** No socket error this
time; the model completed 3 model calls and reported success. But it had
used `write_file` (the only write tool -- full overwrite, no diff/patch)
to replace three real files with tiny illustrative stubs instead of
editing them: `ai_agent/agent.py`'s `RuleBasedAgent` (real keyword-matched
EQ/volume/limiter logic across 5 cases) and `ClaudeAgent` (a real
Anthropic API integration) both became `return {"suggestion": "..."}`
placeholders; `server/main.py` lost `/api/snapshot`, `/api/analyze`, and
`/api/preview` entirely, keeping only a stub `/api/chat`. Test suite went
from 29/29 passing to 5 failing. The model then fabricated a fake `user`
turn mid-response ("Can you also make sure the `Plan` class has a
meaningful representation...") -- no tool call followed, so the harness
correctly treated that as a finished, non-tool response and stopped,
leaving the task both incomplete and destructive. Nothing about the tool
result (`"wrote 810 bytes to ai_agent/agent.py"`) signaled anything was
wrong -- a shrink from a real ~5KB file with working logic to a few
hundred bytes of stub looks, from the tool's return value alone,
identical to a legitimate small edit.

**Recovered cleanly because of a precaution taken *before* this session's
own runs, not because of anything the agent did.** `fl_ai` had no git
history before this project touched it; it was `git init`'d and given a
baseline commit specifically because the existing guidance ("point
`--workdir` only at directories you're fine with a model editing, ideally
under git") had never actually been exercised against a real destructive
failure until now. `git checkout -- <3 files>` restored 29/29 passing
in one command. Without that baseline, this would have been real,
un-undoable damage to someone else's working project.

**Fix, scoped to the actual failure, not a general "agents are risky"
overreaction**: `write_file` now refuses a same-path write that would
shrink an existing file (>=200 bytes) to under 50% of its current size,
returning an error explaining why instead of silently applying it --
`scripts/mini_agent.py`, `tool_write_file`. This is exactly the shape of
what happened both times (a large real file collapsing to a small stub),
cheap to check, doesn't block genuine large deletions written to a
different path, and gives the model a chance to self-correct (read the
full file, preserve what wasn't asked to change) instead of silently
destroying it. It does not fix the deeper cause -- a 14B local model
asked to "modify" a file sometimes writes what it imagines the file
should contain instead of the file it actually read -- only catches the
most damaging shape of that failure after the fact.

**Run 3, with the shrink guard live and an explicit "preserve existing
logic, targeted edit only" instruction added: a third, different failure.**
No stall, no destructive write -- no write at all. The model narrated a
plan in prose that was, this time, actually sound (correctly identified
all three files and a reasonable `format_chat_log` addition preserving
the rest), then called `run_command('python -m pytest -q')` directly
without ever calling `write_file`, got the expected "29 passed" (trivially
true -- nothing had changed), and reported: *"The tests passed
successfully after making the necessary changes... Modified
`ai_agent/context.py`... Modified `ai_agent/agent.py`... Modified
`server/main.py`."* Verified directly: `git status --short` on `fl_ai`
came back completely clean, and none of the three files contain
`chat_log` outside the two pre-existing `state.main.py` append lines.
Zero files touched. The task is exactly as unfinished as before run 1.

**This third failure is arguably worse than the second.** A destructive
overwrite at least changes something a diff or a failing test can catch.
A confident false-success report, corroborated by a real, passing test
run that's meaningless because nothing changed, is the shape of failure
most likely to go unnoticed -- there's no error, no stack trace, no test
failure, just a plausible-sounding summary of work that never happened.

**What three different failures on one well-specified, real, bounded
task actually settles**: the honest update from the earlier agentic-load
correction was "Ramer's placement gap under real agentic load is real
(+28-37%), but that doesn't mean this coding agent is safe or reliable to
use on real code" -- this is the first time that second half was tested
directly rather than assumed, three times, and it failed three different
ways (infra stall / destructive overwrite reported as success / no-op
reported as success). Not a single describable failure pattern to patch
around -- a genuine reliability ceiling on `qwen2.5:14b-instruct` for
open-ended multi-file edits via this tool, at least at this sample size.
This doesn't change anything about Ramer's placement numbers -- all three
runs used the correct cached placement throughout, num_gpu=33 at
ctx=8192, no drift or staleness flagged; the failure is entirely in the
model's tool-use reliability, not in what Ramer optimizes. It changes
what claim "mini_agent.py works on real projects" can honestly support:
it can perform real, correct, single-file edits (the earlier fl_ai
streaming/timeout fixes, and the D:\ramer self-review of the self-healing
fix, both landed cleanly) -- but on a multi-file "modify these existing
files" task, 0 for 3 in this session. Anyone pointed at this tool needs
the git-baseline precaution treated as a requirement, and needs to
actually check the diff, not trust the model's own summary or a passing
test run as proof anything happened. Stopping here rather than retrying
further -- three attempts at the identical task is enough to call this a
real finding, not a fluke worth iterating past.

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

**Update (2026-09-06): Ollama's own GitHub issues independently confirm
Ramer's exact core findings, and revise the "quick upstream PR" assessment
downward.** Searched Ollama's issue tracker and related posts specifically
for VRAM-placement pain, rather than general local-AI frustration. This is
stronger evidence than forum vibes -- these are reproducible engineering
bug reports with hardware specs and logs, not sentiment.

**[ollama/ollama#7629](https://github.com/ollama/ollama/issues/7629)
is close to an exact independent replication of Experiment B**, on
near-identical hardware: RTX 2070 (8GB VRAM, same class as this project's
RTX 5060), Windows, 32GB RAM. Quote: *"llama.cpp achieves 7.2 to 7.5 GB
[utilization]... Ollama only reaches approximately 6GB"* -- the user
explicitly expected Ollama to "use the maximum available VRAM similar to
llama.cpp's behavior" and got slower generation instead. That is, to
within rounding, Experiment B's own measurement (auto: 6.2GB/12.2 tok/s vs.
achievable: 7.83GB/19.6 tok/s). **The important new fact: this issue was
opened November 2024 -- Experiment B reproduced the identical gap
~22 months later**, on different hardware, a different GPU generation, no
visible fix in between. That revises this project's own "buildable in an
afternoon, plausible single upstream PR" assessment of the VRAM-ceiling
gap *downward* in confidence -- if it were that easy, two years and a
specific bug report should have been enough. It's a real, durable,
apparently-not-trivially-fixed gap, not a shallow one about to be
absorbed upstream.

**[ollama/ollama#3837](https://github.com/ollama/ollama/issues/3837)**
independently describes the same *class* of confusion Experiment B found
at the VRAM boundary, on another 8GB card (RTX 3070): *"the GPU vram fills
~7GB but the compute remains at 0-1% and 16 cores of my CPU are active...
the model is running purely on the CPU and not using the GPU"* -- VRAM
being consumed without the expected performance benefit, the same
"filled but not usefully used" confusion as the driver-paging cliff
Experiment B found and confirmed via live `nvidia-smi` monitoring.

**[ollama/ollama#13235](https://github.com/ollama/ollama/issues/13235)
and [#14632](https://github.com/ollama/ollama/issues/14632)** describe
multi-GPU sequential-model-loading failures (three-GPU setups, 24-72GB
total VRAM) -- *"model requires more memory than is currently available,
evicting a model to make space"* followed by unexpected CPU fallback, and
improper same-GPU placement despite other GPUs sitting idle. Different
scale than a single consumer card (closer to H3's "a few GPUs, a small
team" framing than H1/H2), but structurally the same story this project
found and fixed itself: sequential model loading leaves VRAM in a state
the next load doesn't correctly account for -- the exact shape of bug
`unload_others()` fixed for the single-GPU case, evidently unresolved for
Ollama's own multi-GPU placement as of these reports.

**One related but distinct mechanism, worth separating out**: a
[dev.to post](https://dev.to/ljkunal/how-ollama-silently-ate-65gb-of-my-vram-and-how-i-fixed-it-22pf)
describes a 120GB-VRAM production deployment (DGX Spark, continuous
camera-feed inference) where memory crept from ~10GB to 70.8GB because
Ollama pre-allocates KV cache for a model's full *declared* context length
(128K tokens) regardless of actual usage (~1K tokens) -- fixed by manually
capping `num_ctx`. This is KV-cache sizing, not GPU/CPU layer placement --
a different data structure and mechanism (closer to LMCache's territory
per this doc's Competitive Landscape section) -- but the same broader
pattern: Ollama's memory behavior is opaque enough that people build their
own manual workarounds and write it up afterward. Relevant context for
H1/H3 (production/always-on deployments), not something Ramer Local
currently addresses.

**Net effect on the hypotheses**: this is real, corroborating, unprompted
technical evidence (bug reports, not think-pieces) that the exact gap
Ramer Local targets is known, reported, and has persisted for ~2 years
across GPU generations -- meaningfully strengthens H1/H2's "why existing
tools aren't enough" claim specifically for VRAM-placement (separate from
the still-unvalidated "would people pay for it" question, which none of
this speaks to). Still not a substitute for the 20 interviews -- GitHub
issue reporters are also a selection-biased population (people who hit a
bug AND bothered to file it), just a different, more technically rigorous
bias than Reddit sentiment.

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

**Hardware scoping problem, flagged before interviews rather than found
during them (2026-09-06)**: everything built and measured so far --
`ramer_local.py`, both experiments, all of it -- assumes a *discrete*
GPU: VRAM and system RAM are two separate pools connected over PCIe, so
there's a real placement decision and a real cliff at the VRAM boundary.
That's true for RTX 3060/4060/5060-class Windows/Linux machines, the
hardware this project has actually tested on. **It is not true on Apple
Silicon Macs.** Unified memory means CPU and GPU share one RAM pool --
no separate VRAM ceiling, no PCIe hop, nothing to place. Ollama's Metal
backend just loads the model into that shared pool; the binding
constraint there is closer to "does the model fit in unified memory at
all" (a capacity question) than "which layers go in the fast pool vs.
the slow one" (the optimization question this tool answers). Concretely:
`ramer_local.py` calls `nvidia-smi` directly and would simply fail to run
on any Mac today, and a hypothetical Mac port would have little left to
optimize even if built.

This matters for who gets interviewed, not just what gets built:
developers experimenting with local LLMs skew heavily toward Apple
Silicon specifically *because* unified memory (often 32-128GB) makes
larger models fit without a discrete-GPU-style VRAM ceiling at all --
plausibly a large fraction of the exact "individual developer running
local models" population H2 describes. If most interview candidates turn
out to be on Mac, that's not a reason to abandon the interviews; it's a
reason interview question #1 ("what hardware are you running?") should
be used as an early filter, and a reminder that a "yes, this hurts"
answer from a Mac user validates a demand that this specific tool, as
built, cannot address -- a different, unified-memory-shaped problem
(if any) would need its own investigation, not an extension of
`ramer_local.py`.

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

**Where to actually find ~20 people running local AI on constrained
hardware** (the public-research pass above already surfaced these as
real, active communities, not a guess):

- r/LocalLLaMA and r/Ollama (the source of the "8GB VRAM in 2026" and
  MoE-offload threads already cited above -- people there are already
  describing the exact pain points this needs, unprompted, in public)
- r/LocalLLM, with the selection-bias caveat already on record: it's an
  enthusiast bubble, useful for finding people to talk to, not for
  concluding they represent the broader market
- Ollama's own GitHub issues (search "VRAM", "offload", "OOM") --
  people who filed a bug are people who hit the problem hard enough to
  write it up; several such issues are already logged above as
  independent confirmation
- LM Studio's Discord -- an adjacent tool with an overlapping user base,
  useful for checking whether the pain is Ollama-specific or general to
  local inference

**Outreach template** (keep it short; the interview questions above are
the substance, not this message):

> Hey -- I saw your post about [specific thing they said]. I'm looking
> into how people actually run local models day to day -- not selling
> anything, just trying to understand real setups and where they break.
> Would you be up for a 15-minute chat (or even just answering a few
> questions here) about your hardware and workflow?

Anchor every outreach message to something specific they already said
publicly -- it's the difference between a real conversation and a cold
pitch, and it's also a soft filter for people who actually have the
problem rather than people replying to being asked in the abstract.
