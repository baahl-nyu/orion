#!/usr/bin/env python
"""Working set against batch with the GPUs' capacities, and peak throughput per GPU.

    python gpu_ceilings.py [--results DIR]

Four figures under figures/, from the cells under --results (default
~/cheddar-orion-results/final), all Cheddar word32, bins off, managed memory
off, 1 warmup + 3 timed forwards. The cards are the ones in max_qps.py's
CARDS, each with a folder of cells run on that card.

  memory_vs_batch.png            working set of a batch against batch size, one
                                 panel per design, from the RTX PRO 6000 climb
                                 (`ladder/`, ladder.py) including the first
                                 paged batch; the capacity of each GPU is a
                                 horizontal line. The working set does not
                                 depend on the card (max_qps.py checks this),
                                 so one climb gives every card's ceiling.
  memory_vs_batch_one_panel.png  the same on one panel.
  peak_qps_vs_compression.png    peak throughput against compression, one line
                                 per GPU and design, from that GPU's own cells:
                                 the largest batch that ran there without OOM,
                                 divided by its forward on that card. A
                                 configuration whose batch 1 does not fit is a
                                 cross (IVE) or a circled cross (HE-LRM).
  peak_qps_bars.png              the same peak throughput as bars, GPUs on the
                                 x axis, HE-LRM against IVE at three matched
                                 compressions.

The numbers behind the throughput figures are tabulated in max_qps.md; behind
the memory figures, in PAPER_NUMBERS.md (batch ladder).
"""
import argparse
import glob
import json
import os
import sys
from os.path import abspath, dirname, exists, expanduser, join

HERE = dirname(abspath(__file__))
REPO = dirname(dirname(HERE))
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from figstyle import COL_W, FILL, GREY, LINE, RED, apply, ref_line, save, series  # noqa: E402
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
    ap.add_argument("--results", default=expanduser("~/cheddar-orion-results/final"))
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

    def cap_label(ax, name, gib, dy):
        # small cards labelled at the right, where the climbs have left them behind;
        # large cards at the left, where batch 1 sits far below them
        if gib < 60:
            ax.text(34, gib + dy, name, fontsize=5.5, va="bottom", ha="right")
        else:
            ax.text(0.95, gib + dy, name, fontsize=5.5, va="bottom", ha="left")

    cards = sorted(CARDS, key=lambda c: c[2])  # by capacity
    caps = [(f"{round(cap)} GiB ({SHORT[folder]})", cap) for folder, _, cap in cards]
    results = climb(join(args.results, "ladder"))
    cells = {folder: load_cells(join(args.results, folder))[0] for folder, _, _ in cards}

    def best(folder, design, cfg):
        """The largest batch of (design, cfg) that ran on the card, or None."""
        fit = [r for (d, c, b), r in cells[folder].items() if d == design and c == cfg]
        return max(fit, key=lambda r: r["batch"]) if fit else None

    apply()

    def mem_lines(ax, design, ls="-", marker="o"):
        for i, (d, cfg) in enumerate([c for c in CONFIGS if c[0] == design]):
            rows, paged = results[(d, cfg)]
            xs, ys = [r["batch"] for r in rows], [mem(r) for r in rows]
            if paged:
                xs.append(paged["batch"])
                ys.append(mem(paged))
            series(ax, xs, ys, i, label=label(d, cfg), marker=marker, ls=ls)

    def cap_lines(ax):
        for name, gib in caps:
            ref_line(ax, gib, name, left=gib >= 60, dy=2.5)

    # ---- memory_vs_batch: working set against batch, one panel per design, stacked
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for ax, design in zip(axes, ("HE-LRM", "IVE")):
        mem_lines(ax, design)
        cap_lines(ax)
        ax.set_xscale("log", base=2)
        ax.set_ylim(0, 235)  # the top band holds the legend, above the paged points
        ax.set_xlabel("Batch size")
        ax.legend(loc="upper left", ncol=2, title=design, fontsize=6.5)
    axes[0].set_ylabel("Working set (GiB)")
    fig.tight_layout(w_pad=0.8)
    save(fig, "memory_vs_batch")
    plt.close(fig)

    # ---- memory_vs_batch_one_panel: both designs on one panel, HE-LRM solid circles, IVE dashed squares
    fig, ax = plt.subplots(figsize=(COL_W, 3.2))
    mem_lines(ax, "HE-LRM")
    mem_lines(ax, "IVE", ls="--", marker="o")
    cap_lines(ax)
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 275)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Working set (GiB)")
    h, l = ax.get_legend_handles_labels()
    n = len([c for c in CONFIGS if c[0] == "HE-LRM"])
    leg1 = ax.legend(h[:n], l[:n], title="HE-LRM", loc="upper left", fontsize=6, title_fontsize=6.5)
    ax.add_artist(leg1)
    ax.legend(h[n:], l[n:], title="IVE", loc="upper right", fontsize=6, title_fontsize=6.5)
    fig.tight_layout()
    save(fig, "memory_vs_batch_one_panel")
    plt.close(fig)

    # ---- peak_qps_vs_compression: one line per card and design, from the card's own cells
    fig, ax = plt.subplots(figsize=(COL_W, 3.0))
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
                    miss_marker(ax, x, MISS_Y, design, i)
            series(ax, xs, ys, i, marker=marker, ls=ls)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.2, 1000)  # the top decade holds the legend
    ax.set_xlabel("Embedding compression ratio")
    ax.set_ylabel("Peak QPS (largest batch that fits)")
    handles = [Line2D([], [], color=LINE[i], lw=2, label=n) for i, (n, _) in enumerate(caps)]
    handles += [Line2D([], [], color=GREY, ls="-", marker="o", mfc="white", mec="black", label="HE-LRM"),
                Line2D([], [], color=GREY, ls="--", marker="s", mfc="white", mec="black", label="IVE"),
                Line2D([], [], color=GREY, ls="", marker="x", ms=5, mew=1.3, mec=GREY, label="IVE: batch 1 does not fit"),
                (Line2D([], [], color=GREY, ls="", marker="x", ms=5, mew=1.3, mec=GREY),
                 Line2D([], [], mec=GREY, mfc="none", ls="", marker="o", ms=9, mew=0.9))]
    labels = [h.get_label() if not isinstance(h, tuple) else "HE-LRM: batch 1 does not fit" for h in handles]
    ax.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=1, pad=0)}, loc="upper left", ncol=2,
              fontsize=6)
    fig.tight_layout()
    save(fig, "peak_qps_vs_compression")
    plt.close(fig)

    # ---- peak_qps_bars: cards on x, HE-LRM against IVE at matched compression, one row (two-column width)
    import numpy as np
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
        "memory_vs_batch.png", "memory_vs_batch_one_panel.png", "peak_qps_vs_compression.png", "peak_qps_bars.png")))


if __name__ == "__main__":
    main()
