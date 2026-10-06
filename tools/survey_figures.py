"""Generate the figures for docs/Ramer_Literature_Survey.pdf.

Every number plotted here is a real measurement recorded in this repo
(PRODUCT_HYPOTHESES.md / docs/ARCHITECTURE.md / results/). Nothing is
synthetic, interpolated, or smoothed.

Palette: the validated categorical default (slots 1-3 only, which clear the
all-pairs CVD floors), text in ink tokens rather than series colors, direct
labels everywhere so identity is never carried by color alone.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# --- design tokens -------------------------------------------------------
S1 = "#2a78d6"   # blue    - "Ramer" / optimized
S2 = "#eb6834"   # orange  - "baseline" / auto
S3 = "#1baf7a"   # aqua    - third series when needed
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e3e2de"
SURFACE = "#ffffff"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.5,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "legend.frameon": False,
    "legend.fontsize": 8,
})


def tidy(ax, xgrid=False):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="x" if xgrid else "y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("wrote", path)


# =========================================================================
# Figure 1 - taxonomy tree
# =========================================================================
def fig1_taxonomy():
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.set_xlim(0, 104)
    ax.set_ylim(-10, 100)
    ax.axis("off")

    families = [
        ("F1  Model compression",
         "quantization (GPTQ, AWQ,\nK/IQ-quants), pruning,\ndistillation, low-rank"),
        ("F2  Decoding algorithms",
         "speculative decoding,\ndraft models, early exit,\nsparse activation"),
        ("F3  KV-cache management",
         "PagedAttention, eviction\n(H2O, StreamingLLM), KV\nquantization, LMCache tiering"),
        ("F4  Weight placement\n       and offloading",
         "ZeRO-Inference, FlexGen,\nLLM in a flash, PowerInfer,\nMoE offload, llama.cpp -ngl"),
        ("F5  Batching and request\n       scheduling",
         "Orca iteration-level batching,\nSarathi-Serve chunked prefill,\ncontinuous batching"),
        ("F6  Multi-model resource\n       coordination",
         "Prism / kvcached ballooning,\nServerlessLLM loading and\nplacement, model residency"),
    ]

    # root
    ax.add_patch(FancyBboxPatch((26, 86), 48, 10, boxstyle="round,pad=0.6,rounding_size=1.6",
                                linewidth=1.4, edgecolor=S1, facecolor="#eaf2fc", zorder=3))
    ax.text(50, 91, "Local LLM inference optimization", ha="center", va="center",
            fontsize=10, fontweight="bold", color=INK, zorder=4)

    y0 = 76
    dy = 13.2
    for i, (title, leaves) in enumerate(families):
        y = y0 - i * dy
        ax.add_patch(FancyBboxPatch((4, y - 5.0), 34.5, 10.0,
                                    boxstyle="round,pad=0.5,rounding_size=1.4",
                                    linewidth=1.1, edgecolor=S1, facecolor="#f6f9fd", zorder=3))
        ax.text(6.0, y, title, ha="left", va="center", fontsize=8.4,
                fontweight="bold", color=INK, zorder=4)
        # connector from root spine
        ax.plot([25, 25, 4], [86, y, y], color=MUTED, linewidth=0.9, zorder=1,
                solid_capstyle="round")
        ax.plot([39.5, 44], [y, y], color=MUTED, linewidth=0.9, zorder=1)
        ax.text(45, y, leaves, ha="left", va="center", fontsize=7.6, color=INK2, zorder=4,
                linespacing=1.35)

    ax.plot([25, 50], [86, 86], color=MUTED, linewidth=0.9, zorder=1)

    ax.text(50, -9.0,
            "Cross-cutting: configuration auto-tuning (llama.cpp --fit, llm-optimizer, Bench360),\n"
            "which searches over the levers above rather than adding a new one.",
            ha="center", va="bottom", fontsize=7.6, color=MUTED, linespacing=1.35)
    save(fig, "fig1_taxonomy.png")


# =========================================================================
# Figure 2 - offload sweep (Experiment B)
# =========================================================================
def fig2_offload_sweep():
    layers = [0, 8, 16, 24, 32, 40, 48]
    decode = [5.11, 6.18, 7.27, 9.09, 12.86, 19.64, 6.17]
    vram = [0.0, 2.12, 3.53, 4.96, 6.40, 7.83, 9.30]
    auto_tps, auto_vram = 12.21, 6.20

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.9, 4.6), sharex=True,
                                   gridspec_kw={"height_ratios": [1.5, 1.0], "hspace": 0.16})

    ax1.plot(layers[:6], decode[:6], color=S1, linewidth=2.0, marker="o",
             markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=4)
    ax1.plot(layers[5:], decode[5:], color=S2, linewidth=2.0, linestyle=(0, (4, 2)),
             marker="o", markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=4)
    ax1.plot([48], [4.40], marker="o", markersize=5.5, color=S2,
             markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=4)
    ax1.axhline(auto_tps, color=MUTED, linewidth=1.3, linestyle=(0, (2, 2)), zorder=2)

    ax1.annotate("19.64 tok/s measured optimum\n(40 layers, 7.83 GB)",
                 xy=(40, 19.64), xytext=(25.0, 21.0), fontsize=7.8, color=INK,
                 ha="center", va="bottom", linespacing=1.3)
    ax1.annotate("Ollama auto default: 12.21 tok/s",
                 xy=(2, auto_tps), xytext=(1.5, 12.9), fontsize=7.8, color=INK2)
    ax1.annotate("oversubscription cliff: 6.17 and 4.40 tok/s\n"
                 "on two identical runs, below the all-CPU baseline",
                 xy=(47.4, 4.9), xytext=(17.0, 1.1), fontsize=7.8, color=INK,
                 ha="left", va="bottom", linespacing=1.3,
                 arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.9,
                                 connectionstyle="arc3,rad=-0.12"))
    ax1.set_ylabel("decode throughput (tok/s)")
    ax1.set_title("Qwen2.5-14B Q4_K_M on an RTX 5060 8 GB: throughput against GPU layer count",
                  fontsize=9)
    ax1.set_ylim(0, 25.5)
    tidy(ax1)

    ax2.bar(layers, vram, width=3.6, color=S1, zorder=3)
    ax2.axhline(8.15, color=S2, linewidth=1.4, linestyle=(0, (3, 2)), zorder=4)
    ax2.text(1.0, 8.5, "physical VRAM 8.15 GB", fontsize=7.8, color=S2)
    ax2.set_ylabel("VRAM requested (GB)")
    ax2.set_xlabel("transformer layers placed on GPU (of 48)")
    ax2.set_ylim(0, 11)
    ax2.set_xticks(layers)
    tidy(ax2)

    save(fig, "fig2_offload_sweep.png")


# =========================================================================
# Figure 3 - placement gain across three models
# =========================================================================
def fig3_placement_gain():
    models = ["Qwen2.5-14B\nQ4_K_M", "Qwen2.5-32B\nQ2_K", "Qwen3.8-27B\nIQ2_XXS"]
    auto = [12.4, 3.1, 5.9]
    ramer = [19.6, 3.3, 29.6]
    gains = ["+61%", "+6.7%", "+402%"]

    x = range(len(models))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    b1 = ax.bar([i - w / 2 - 0.012 for i in x], auto, width=w, color=S2, zorder=3,
                label="Ollama auto placement")
    b2 = ax.bar([i + w / 2 + 0.012 for i in x], ramer, width=w, color=S1, zorder=3,
                label="Ramer measured placement")

    for rects in (b1, b2):
        for r in rects:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.6,
                    f"{r.get_height():.1f}", ha="center", va="bottom",
                    fontsize=7.8, color=INK2)
    for i, g in enumerate(gains):
        ax.text(i, max(auto[i], ramer[i]) + 3.0, g, ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=INK)

    ax.set_xticks(list(x))
    ax.set_xticklabels(models)
    ax.set_ylabel("decode throughput (tok/s)")
    ax.set_ylim(0, 36)
    ax.set_title("Measured placement against the engine's own default, three models, one machine")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0))
    tidy(ax)
    save(fig, "fig3_placement_gain.png")


# =========================================================================
# Figure 4 - context sweep (Experiment C)
# =========================================================================
def fig4_context_sweep():
    ctx = [2048, 4096, 6144, 8192, 12288]
    auto = [12.70, 12.96, 12.07, 11.64, 10.11]
    best = [16.64, 16.55, 14.02, 13.98, 13.70]

    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ax.plot(ctx, best, color=S1, linewidth=2.0, marker="o", markersize=5.5,
            markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=4)
    ax.plot(ctx, auto, color=S2, linewidth=2.0, marker="o", markersize=5.5,
            markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=4)

    ax.text(12500, best[-1], "  Ramer, re-probed\n  at each context",
            color=INK, fontsize=8, va="center", linespacing=1.3)
    ax.text(12500, auto[-1], "  Ollama auto", color=INK, fontsize=8, va="center")
    for cx, a, b in zip(ctx, auto, best):
        ax.text(cx, b + 0.45, f"{b:.1f}", ha="center", fontsize=7.4, color=INK2)
        ax.text(cx, a - 1.0, f"{a:.1f}", ha="center", fontsize=7.4, color=INK2)

    ax.set_xticks(ctx)
    ax.set_xticklabels([str(c) for c in ctx])
    ax.set_xlabel("context window (tokens)")
    ax.set_ylabel("decode throughput (tok/s)")
    ax.set_xlim(1200, 17200)
    ax.set_ylim(8, 19)
    ax.set_title("Placement validity is context-dependent: the same model, the same GPU, only num_ctx varies")
    tidy(ax)
    save(fig, "fig4_context_sweep.png")


# =========================================================================
# Figure 5 - scheduler ablation
# =========================================================================
def fig5_scheduler():
    conds = ["A  no coordination", "B  sticky affinity only",
             "C  priority only", "D  sticky + priority"]
    median = [210.6, 137.7, 195.2, 117.6]
    switches = [6.6, 3.4, 6.0, 3.6]
    y = list(range(len(conds)))[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9),
                                   gridspec_kw={"width_ratios": [1.5, 1.0], "wspace": 0.12})

    ax1.barh(y, median, height=0.58, color=[S2, S1, S2, S1], zorder=3)
    for yy, v in zip(y, median):
        ax1.text(v + 4, yy, f"{v:.1f}s", va="center", fontsize=7.8, color=INK2)
    ax1.set_yticks(y)
    ax1.set_yticklabels(conds, fontsize=8)
    ax1.set_xlabel("median wall time, 4 concurrent agents (s)")
    ax1.set_xlim(0, 250)
    ax1.set_title("Time to finish", loc="left")
    tidy(ax1, xgrid=True)

    ax2.barh(y, switches, height=0.58, color=[S2, S1, S2, S1], zorder=3)
    for yy, v in zip(y, switches):
        ax2.text(v + 0.12, yy, f"{v:.1f}", va="center", fontsize=7.8, color=INK2)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlabel("mean model switches per trial")
    ax2.set_xlim(0, 8.2)
    ax2.set_title("Thrashing", loc="left")
    tidy(ax2, xgrid=True)

    save(fig, "fig5_scheduler.png")


# =========================================================================
# Figure 6 - end-to-end agentic A/B
# =========================================================================
def fig6_endtoend():
    labels = ["Coding task, 14B Q4_K_M\n(both runs correct)",
              "Research task, 14B Q4_K_M\n(web search + read)"]
    without = [89.2, 19.9]
    with_r = [55.4, 19.8]
    y = [1, 0]
    h = 0.32

    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    ax.barh([yy + h / 2 + 0.012 for yy in y], without, height=h, color=S2, zorder=3,
            label="agent pointed at Ollama directly")
    ax.barh([yy - h / 2 - 0.012 for yy in y], with_r, height=h, color=S1, zorder=3,
            label="agent pointed at Ramer")
    for yy, a, b in zip(y, without, with_r):
        ax.text(a + 1.5, yy + h / 2 + 0.012, f"{a:.1f}s", va="center", fontsize=7.8, color=INK2)
        ax.text(b + 1.5, yy - h / 2 - 0.012, f"{b:.1f}s", va="center", fontsize=7.8, color=INK2)
    ax.text(97, 1.0, "38%\nfaster", fontsize=8.6, fontweight="bold", color=INK,
            va="center", ha="center", linespacing=1.3)
    ax.text(26, 0.0, "no measurable difference", fontsize=8.6, color=INK, va="center")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("end-to-end task wall time, pi CLI agent (s), lower is better")
    ax.set_xlim(0, 112)
    ax.set_ylim(-0.85, 1.6)
    ax.set_title("The benefit tracks how much of the task is local decoding")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, -0.02))
    tidy(ax, xgrid=True)
    save(fig, "fig6_endtoend.png")


# =========================================================================
# Figure 7 - tiering strategy ablation (research phase)
# =========================================================================
def fig7_tiering():
    labels = ["Synchronous\nfetch on demand", "+ next-layer\nprefetch",
              "+ persistent\nRAM cache"]
    vals = [1.6, 1.55, 15.5]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.bar(range(3), vals, width=0.5, color=[MUTED, MUTED, S1], zorder=3)
    ax.axhline(31, color=S2, linewidth=1.4, linestyle=(0, (3, 2)), zorder=4)
    ax.text(-0.42, 31.9, "fully GPU-resident baseline: ~31 tok/s", fontsize=7.8, color=S2)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.8, f"{v:.1f}", ha="center", fontsize=8.2, color=INK2)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("decode throughput (tok/s)")
    ax.set_ylim(0, 36)
    ax.set_title("Caching, not prefetching, is what recovers a remote/offloaded tier")
    tidy(ax)
    save(fig, "fig7_tiering.png")


if __name__ == "__main__":
    fig1_taxonomy()
    fig2_offload_sweep()
    fig3_placement_gain()
    fig4_context_sweep()
    fig5_scheduler()
    fig6_endtoend()
    fig7_tiering()
    print("all figures written to", OUT)
