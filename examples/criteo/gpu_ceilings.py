#!/usr/bin/env python
"""Working set against batch with the GPUs' capacities, and peak throughput per GPU.

    python gpu_ceilings.py [--results DIR]

Three figures under figures/, from the cells under --results (default
examples/criteo/results), all Cheddar word32, bins off, managed memory
off, 1 warmup + 3 timed forwards. The cards are the ones in max_qps.py's
CARDS, each with a folder of cells run on that card.

  memory_vs_batch.png            working set of a batch against batch size, both
                                 designs on one panel (HE-LRM circles just left
                                 of each batch, IVE squares just right), from the
                                 RTX PRO 6000 climb (`ladder/`, ladder.py)
                                 including the first paged batch; the capacity
                                 of each GPU is a horizontal line. The working
                                 set does not depend on the card (max_qps.py
                                 checks this), so one climb gives every card's
                                 ceiling.
  peak_qps_vs_compression.png    peak throughput against compression, one line
                                 per GPU and design, from that GPU's own cells:
                                 the largest batch that ran there without OOM,
                                 divided by its forward on that card. A
                                 configuration whose batch 1 does not fit is
                                 its own marker on the 0 line.
  peak_qps_bars.png              the same peak throughput as bars, GPUs on the
                                 x axis, HE-LRM against IVE at three matched
                                 compressions.

The numbers behind the throughput figures are tabulated in max_qps.md; behind
the memory figures, in PAPER_NUMBERS.md (batch ladder).
"""
import argparse
import glob
import json
import math
import os
import sys
from os.path import abspath, dirname, exists, expanduser, join

HERE = dirname(abspath(__file__))
REPO = dirname(dirname(HERE))
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from figstyle import COL_W, FILL, GREY, LINE, RED, apply, save, series  # noqa: E402
from ladder import CONFIGS, IVE_COMPRESSION, row_from, tag  # noqa: E402
from max_qps import CARDS, load_cells  # noqa: E402
from table2 import fmt_compression  # noqa: E402

FIG = join(HERE, "figures")
BATCHES = (1, 2, 4, 8, 16, 32)
# short card names, keyed by results folder
SHORT = {"ladder": "RTX PRO 6000", "ladder_a100": "A100", "ladder_3090": "RTX 3090", "ladder_h100": "H100"}
MISS_Y = 0.35  # where a configuration that does not fit at batch 1 is marked
# HE-LRM against IVE at the closest compressions the two parameter sets share
PAIRS = [("~700x", ("HE-LRM", 50000), ("IVE", 1024)),
         ("~4,000x", ("HE-LRM", 5000), ("IVE", 128)),
         ("highest", ("HE-LRM", 500), ("IVE", 32))]


def climb(root):
    """{(design, cfg): (rows, paged)} of the RTX PRO 6000 climb, rows ordered by batch."""
    out = {}
    for design, cfg in CONFIGS:
        rows, paged = [], None
        for b in BATCHES:
            for mm in (False, True):
                name = tag(design, cfg, b, mm)
                js = glob.glob(join(root, name, "*.json")) if design == "HE-LRM" else \
                    ([join(root, name + ".json")] if exists(join(root, name + ".json")) else [])
                if not js:
                    continue
                row = row_from(design, b, mm, json.load(open(js[0])))
                if mm:
                    paged = row
                else:
                    rows.append(row)
        out[(design, cfg)] = (rows, paged)
    return out


def miss_marker(ax, x, y, design, i, ms=5):
    """Batch 1 does not fit: IVE a cross, HE-LRM a cross in a circle, in the card's line colour."""
    ax.plot([x], [y], "x", color=LINE[i], ms=ms, mew=1.3, mec=LINE[i], zorder=3)
    if design == "HE-LRM":
        ax.plot([x], [y], "o", mfc="none", mec=LINE[i], ms=ms + 4, mew=0.9, zorder=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=join(HERE, "results"))
    args = ap.parse_args()
    from bench_sweep import SPARSE_SIZES
    from dlrm import expanded_width
    total = sum(SPARSE_SIZES)

    def compression(design, cfg):
        if design == "HE-LRM":
            return total / sum(expanded_width(s, cfg, 4) for s in SPARSE_SIZES)
        return IVE_COMPRESSION[cfg]

    def label(design, cfg):
        return (f"t={cfg:,}" if design == "HE-LRM" else f"p={cfg}") + f", {fmt_compression(compression(design, cfg))}x"

    def mem(row):
        return row["live"] if row["live"] else row["peak"]

    cards = sorted(CARDS, key=lambda c: c[2])  # by capacity
    caps = [(f"{round(cap)} GiB ({SHORT[folder]})", cap) for folder, _, cap in cards]
    results = climb(join(args.results, "ladder"))
    cells = {folder: load_cells(join(args.results, folder))[0] for folder, _, _ in cards}

    def best(folder, design, cfg):
        """The largest batch of (design, cfg) that ran on the card, or None."""
        fit = [r for (d, c, b), r in cells[folder].items() if d == design and c == cfg]
        return max(fit, key=lambda r: r["batch"]) if fit else None

    apply()

    # ---- memory_vs_batch: working set against batch, both designs on one column-width panel.
    # Markers only: HE-LRM circles just left of each batch, IVE squares just right, a brace under
    # each pair carrying the batch label. y is linear to 50 GiB, 2x compressed to 100 GiB so the
    # 80 and 96 GiB lines stay apart, 3x to 150 and 10x above; the top band holds the legends.
    import math
    from matplotlib.ticker import FixedLocator, NullFormatter

    def short_t(t):
        return f"{t:,}" if t < 1000 else (f"{t // 1000}k" if t < 1_000_000 else f"{t // 1_000_000}M")

    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    DODGE = 0.09  # log2 units either side of the batch
    for design, marker, sign in (("HE-LRM", "o", -1), ("IVE", "s", 1)):
        for i, (d, cfg) in enumerate([c for c in CONFIGS if c[0] == design]):
            rows, paged = results[(d, cfg)]
            xs, ys = [r["batch"] * 2 ** (sign * DODGE) for r in rows], [mem(r) for r in rows]
            if paged:
                xs.append(paged["batch"] * 2 ** (sign * DODGE))
                ys.append(mem(paged))
            ax.plot(xs, ys, marker, mfc=FILL[i], mec="black", ls="none", ms=4, zorder=3,
                    label=f"t={short_t(cfg)}" if d == "HE-LRM" else f"p={cfg}")
    for folder, _, gib in cards:  # capacity lines: 96 a bit over its line and 80 under it at the left, 40 and 24 at the right
        ax.axhline(gib, color=RED, lw=1.1, ls="--", zorder=1.6)
        left = gib >= 60
        below = 60 <= gib < 90
        ax.text(0.015 if left else 0.985, gib + (-3.5 if below else 3.0 if gib >= 90 else 1.5),
                f"{round(gib)} GiB ({SHORT[folder]})", fontsize=5.3, color=RED, fontweight="bold",
                va="top" if below else "bottom", ha="left" if left else "right",
                transform=ax.get_yaxis_transform(), zorder=4)
    # knees and slopes of the piecewise-linear y axis: 1x to 50 GiB, 1/2 to 100, 1/3 to 150, 1/10 above
    KNEES, SLOPES = [50.0, 100.0, 150.0], [1.0, 1 / 2, 1 / 3, 1 / 10]

    def fw(y):
        y = np.asarray(y, dtype=float)
        u, lo, base = np.zeros_like(y), 0.0, 0.0
        for knee, k in zip(KNEES + [np.inf], SLOPES):
            u = np.where((y > lo) & (y <= knee), base + (y - lo) * k, u)
            if np.isfinite(knee):
                base, lo = base + (knee - lo) * k, knee
        return np.where(y <= 0, y, u)

    def iv(u):
        u = np.asarray(u, dtype=float)
        y, lo, base = np.zeros_like(u), 0.0, 0.0
        for knee, k in zip(KNEES + [np.inf], SLOPES):
            top = base + (knee - lo) * k if np.isfinite(knee) else np.inf
            y = np.where((u > base) & (u <= top), lo + (u - base) / k, y)
            if np.isfinite(knee):
                base, lo = top, knee
        return np.where(u <= 0, u, y)

    ax.set_yscale("function", functions=(fw, iv))
    ax.set_ylim(10, 230)
    ax.yaxis.set_major_locator(FixedLocator([10, 20, 30, 40, 50, 100, 200]))
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(list(BATCHES)))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.tick_params(axis="x", which="both", length=0)
    ax.set_ylabel("Working set (GiB)")
    ax.set_xlabel("Batch size", labelpad=15)
    h, l = ax.get_legend_handles_labels()
    n = len([c for c in CONFIGS if c[0] == "HE-LRM"])
    kw = dict(fontsize=5.3, borderpad=0.25, labelspacing=0.15, handletextpad=0.3, handlelength=1.0, columnspacing=0.5)
    leg1 = ax.legend(h[:n], l[:n], loc="upper left", ncol=3, **kw)  # two rows each, side by side in the top band
    ax.add_artist(leg1)
    fig.canvas.draw()
    bb = leg1.get_window_extent().transformed(ax.transAxes.inverted())
    ax.legend(h[n:], l[n:], loc="upper left", ncol=3, bbox_to_anchor=(bb.x1 + 0.015, 1.0), **kw)
    # braces under each batch, hugging the two columns, batch label beneath
    l0, l1 = (math.log2(v) for v in ax.get_xlim())
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    for b in BATCHES:
        fa, fb = (((math.log2(b) + e * 0.17) - l0) / (l1 - l0) for e in (-1, 1))
        u = np.linspace(0, 1, 301)
        half = 0.5 * (1 - sig(90 * (u - 0.06))) + 0.5 * (1 - sig(90 * (u - 0.44)))
        y = np.where(u <= 0.5, half, half[::-1])  # arms at the ends, cusp in the middle
        ax.plot(fa + (fb - fa) * u, -0.06 + 0.045 * y, color="0.2", lw=0.6, transform=ax.transAxes, clip_on=False, zorder=5)
        ax.text((math.log2(b) - l0) / (l1 - l0), -0.09, f"$2^{{{int(math.log2(b))}}}$", ha="center", va="top",
                fontsize=7.5, transform=ax.transAxes)
    fig.tight_layout()
    save(fig, "memory_vs_batch")
    plt.close(fig)

    # ---- peak_qps_vs_compression: one line per card and design, from the card's own cells.
    # A configuration whose batch 1 does not fit on the card is its own marker on a "0" line
    # (drawn at MISS_Y on the log axis, tick relabelled 0).
    from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    for i, (folder, _, cap) in enumerate(cards):
        for design, ls, marker in (("HE-LRM", "-", "o"), ("IVE", "--", "s")):
            xs, ys = [], []
            for d, cfg in CONFIGS:
                if d != design:
                    continue
                x = compression(d, cfg)
                r = best(folder, d, cfg)
                if r:
                    xs.append(x)
                    ys.append(r["qps"])
                else:
                    ax.plot([x], [MISS_Y], marker, mfc=FILL[i], mec="black", ls="none", ms=4.5, zorder=3)
            series(ax, xs, ys, i, marker=marker, ls=ls, ms=4.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.2, 700)  # the top band holds the legend
    ax.yaxis.set_major_locator(FixedLocator([MISS_Y, 1, 10, 100]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == MISS_Y else f"$10^{{{round(math.log10(v))}}}$"))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10), numticks=20))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.axhline(MISS_Y, color="0.5", lw=0.8, ls=":", zorder=1)
    ax.set_xlabel("Embedding compression ratio")
    ax.set_ylabel("Peak QPS")
    card_h = [Line2D([], [], color=LINE[i], lw=2, label=n) for i, (n, _) in enumerate(caps)]
    helrm_h = Line2D([], [], color=GREY, ls="-", marker="o", mfc="white", mec="black", ms=4.5, label="HE-LRM")
    ive_h = Line2D([], [], color=GREY, ls="--", marker="s", mfc="white", mec="black", ms=4.5, label="IVE")
    # two columns of three, filled column-wise: the two smaller cards then HE-LRM, the two larger then IVE
    handles = card_h[:2] + [helrm_h] + card_h[2:] + [ive_h]
    ax.legend(handles=handles, loc="upper left", ncol=2, fontsize=5.5, borderpad=0.3, labelspacing=0.25,
              handletextpad=0.4, handlelength=1.4, columnspacing=0.8)
    fig.tight_layout()
    save(fig, "peak_qps_vs_compression")
    plt.close(fig)

    # ---- peak_qps_bars: cards on x, HE-LRM against IVE at matched compression, one row (two-column width)
    fig, axes = plt.subplots(1, len(PAIRS), figsize=(7.0, 2.4), sharey=True)
    ticks = [f"{round(cap)}\n{SHORT[folder].replace('RTX PRO 6000', 'PRO 6000')}" for folder, _, cap in cards]
    xs = np.arange(len(cards))
    w = 0.38
    top = 0
    for ax, (title, a, b) in zip(axes, PAIRS):
        for off, key, i in ((-w / 2, a, 0), (w / 2, b, 1)):
            vals, bs = [], []
            for folder, _, _ in cards:
                r = best(folder, *key)
                vals.append(r["qps"] if r else 0)
                bs.append(r["batch"] if r else None)
            top = max(top, max(vals))
            ax.bar(xs + off, vals, w, color=FILL[i], edgecolor="black", linewidth=0.7,
                   label=f"{key[0]} ({label(*key)})", zorder=2)
            for x, v, bb in zip(xs + off, vals, bs):
                if bb:
                    ax.text(x, v + 1, f"b{bb}", fontsize=6, ha="center", va="bottom", color=LINE[i])
                else:
                    miss_marker(ax, x, 1.5, key[0], i, ms=4)
        ax.set_xticks(xs)
        ax.set_xticklabels(ticks, fontsize=7)
        ax.set_title(f"compression {title}")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left", fontsize=6)
    axes[0].set_ylabel("Peak QPS (largest batch that fits)")
    axes[0].set_ylim(0, top * 1.22)
    fig.supxlabel("GPU memory (GiB)", fontsize=8.5, y=0.02)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.27)
    save(fig, "peak_qps_bars")
    plt.close(fig)
    print("wrote " + ", ".join(join(FIG, f) for f in (
        "memory_vs_batch.png", "peak_qps_vs_compression.png", "peak_qps_bars.png")))


if __name__ == "__main__":
    main()
