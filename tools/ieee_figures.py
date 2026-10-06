"""Column-width figure variants for docs/Ramer_IEEE_Paper.pdf.

Same measurements as tools/survey_figures.py, re-laid-out for a 3.5 in
IEEE column so the type stays legible at final size.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e3e2de"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 6.2,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 6.8, "axes.titleweight": "bold", "axes.labelsize": 6.2,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 5.9, "ytick.labelsize": 5.9,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "legend.frameon": False, "legend.fontsize": 5.9,
})
W = 3.42


def tidy(ax, xgrid=False):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="x" if xgrid else "y", color=GRID, linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def save(fig, name):
    fig.savefig(OUT / name, dpi=320, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("wrote", OUT / name)


def f_offload():
    lay = [0, 8, 16, 24, 32, 40, 48]
    dec = [5.11, 6.18, 7.27, 9.09, 12.86, 19.64, 6.17]
    fig, ax = plt.subplots(figsize=(W, 2.35))
    ax.plot(lay[:6], dec[:6], color=S1, lw=1.5, marker="o", ms=3.6,
            markeredgecolor="white", markeredgewidth=1.0, zorder=4)
    ax.plot(lay[5:], dec[5:], color=S2, lw=1.5, ls=(0, (3, 1.6)), marker="o", ms=3.6,
            markeredgecolor="white", markeredgewidth=1.0, zorder=4)
    ax.plot([48], [4.40], "o", ms=3.6, color=S2, markeredgecolor="white",
            markeredgewidth=1.0, zorder=4)
    ax.axhline(12.21, color=MUTED, lw=1.0, ls=(0, (2, 2)), zorder=2)
    ax.text(0.5, 12.9, "engine auto: 12.21", fontsize=5.9, color=INK2)
    ax.annotate("19.64 optimum\n(7.83 GB, fits)", xy=(40, 19.64), xytext=(21, 20.6),
                fontsize=5.9, color=INK, ha="center", linespacing=1.25)
    ax.annotate("9.30 GB requested on an\n8.15 GB card: 6.17 and 4.40",
                xy=(47.2, 4.9), xytext=(15.5, 1.0), fontsize=5.9, color=INK,
                linespacing=1.25,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7,
                                connectionstyle="arc3,rad=-0.12"))
    ax.set_xlabel("GPU layers (of 48)")
    ax.set_ylabel("decode tok/s")
    ax.set_xticks(lay)
    ax.set_ylim(0, 25)
    tidy(ax)
    save(fig, "ieee_offload.png")


def f_gain():
    models = ["14B\nQ4_K_M", "32B\nQ2_K", "27B\nIQ2_XXS"]
    auto, ram = [12.2, 3.1, 5.9], [19.6, 3.3, 29.6]
    gains = ["+61%", "+6.7%", "+402%"]
    x, w = range(3), 0.35
    fig, ax = plt.subplots(figsize=(W, 2.1))
    b1 = ax.bar([i - w / 2 - 0.01 for i in x], auto, w, color=S2, zorder=3, label="engine auto")
    b2 = ax.bar([i + w / 2 + 0.01 for i in x], ram, w, color=S1, zorder=3, label="measured")
    for rs in (b1, b2):
        for r in rs:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.5,
                    f"{r.get_height():.1f}", ha="center", fontsize=5.6, color=INK2)
    for i, g in enumerate(gains):
        ax.text(i, max(auto[i], ram[i]) + 3.0, g, ha="center", fontsize=6.6,
                fontweight="bold", color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels(models)
    ax.set_ylabel("decode tok/s")
    ax.set_ylim(0, 37)
    ax.legend(loc="upper left")
    tidy(ax)
    save(fig, "ieee_gain.png")


def f_ctx():
    ctx = [2048, 4096, 6144, 8192, 12288]
    auto = [12.70, 12.96, 12.07, 11.64, 10.11]
    best = [16.64, 16.55, 14.02, 13.98, 13.70]
    fig, ax = plt.subplots(figsize=(W, 2.0))
    ax.plot(ctx, best, color=S1, lw=1.5, marker="o", ms=3.4,
            markeredgecolor="white", markeredgewidth=1.0)
    ax.plot(ctx, auto, color=S2, lw=1.5, marker="o", ms=3.4,
            markeredgecolor="white", markeredgewidth=1.0)
    ax.text(4300, 17.2, "re-probed at each context", color=INK, fontsize=5.9)
    ax.text(4300, 11.0, "engine auto", color=INK, fontsize=5.9)
    ax.set_xticks(ctx)
    ax.set_xticklabels(["2k", "4k", "6k", "8k", "12k"])
    ax.set_xlabel("context window (tokens)")
    ax.set_ylabel("decode tok/s")
    ax.set_ylim(9, 18.6)
    tidy(ax)
    save(fig, "ieee_ctx.png")


def f_sched():
    conds = ["A no coordination", "B sticky only", "C priority only", "D sticky+priority"]
    med = [210.6, 137.7, 195.2, 117.6]
    sw = [6.6, 3.4, 6.0, 3.6]
    y = list(range(4))[::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 1.75),
                                 gridspec_kw={"width_ratios": [1.7, 1.0], "wspace": 0.1})
    a1.barh(y, med, 0.56, color=[S2, S1, S2, S1], zorder=3)
    for yy, v in zip(y, med):
        a1.text(v + 5, yy, f"{v:.0f}", va="center", fontsize=5.6, color=INK2)
    a1.set_yticks(y)
    a1.set_yticklabels(conds, fontsize=5.9)
    a1.set_xlabel("median wall time (s)")
    a1.set_xlim(0, 260)
    tidy(a1, xgrid=True)
    a2.barh(y, sw, 0.56, color=[S2, S1, S2, S1], zorder=3)
    for yy, v in zip(y, sw):
        a2.text(v + 0.15, yy, f"{v:.1f}", va="center", fontsize=5.6, color=INK2)
    a2.set_yticks(y)
    a2.set_yticklabels([])
    a2.set_xlabel("model switches")
    a2.set_xlim(0, 8.6)
    tidy(a2, xgrid=True)
    save(fig, "ieee_sched.png")


def f_e2e():
    lab = ["coding, 14B\n(both correct)", "research, 14B\n(web tools)"]
    wo, wi = [89.2, 19.9], [55.4, 19.8]
    y, h = [1, 0], 0.3
    fig, ax = plt.subplots(figsize=(W, 1.7))
    ax.barh([v + h / 2 + 0.01 for v in y], wo, h, color=S2, zorder=3, label="direct to engine")
    ax.barh([v - h / 2 - 0.01 for v in y], wi, h, color=S1, zorder=3, label="through Ramer")
    for yy, a, b in zip(y, wo, wi):
        ax.text(a + 1.5, yy + h / 2 + 0.01, f"{a:.1f}", va="center", fontsize=5.6, color=INK2)
        ax.text(b + 1.5, yy - h / 2 - 0.01, f"{b:.1f}", va="center", fontsize=5.6, color=INK2)
    ax.text(62, 1.0, "38% faster", fontsize=6.2, fontweight="bold", color=INK, va="center")
    ax.text(26, 0.0, "no difference", fontsize=6.2, color=INK, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(lab, fontsize=5.9)
    ax.set_xlabel("task wall time (s)")
    ax.set_xlim(0, 108)
    ax.set_ylim(-0.95, 1.55)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, -0.06))
    tidy(ax, xgrid=True)
    save(fig, "ieee_e2e.png")


if __name__ == "__main__":
    f_offload(); f_gain(); f_ctx(); f_sched(); f_e2e()
