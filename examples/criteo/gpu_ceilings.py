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

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from ladder import CONFIGS, IVE_COMPRESSION, row_from, tag  # noqa: E402
from max_qps import CARDS, load_cells  # noqa: E402
from table2 import fmt_compression  # noqa: E402

FIG = join(HERE, "figures")
BATCHES = (1, 2, 4, 8, 16, 32)
# short card names, keyed by results folder
SHORT = {"ladder": "RTX PRO 6000", "ladder_a100": "A100", "ladder_3090": "RTX 3090", "ladder_h100": "H100"}
# one colour per card, smallest to largest
CARD_PAL = ["#222222", "#1f78b4", "#2ca25f", "#e6550d"]
HELRM_C, IVE_C = "#2a78d6", "#eb6834"
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


def miss_marker(ax, x, y, design, color, ms=5):
    """Batch 1 does not fit: IVE a cross, HE-LRM a cross in a circle."""
    ax.plot([x], [y], "x", color=color, ms=ms, mew=1.2)
    if design == "HE-LRM":
        ax.plot([x], [y], "o", mfc="none", mec=color, ms=ms + 4, mew=0.9)


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

    os.makedirs(FIG, exist_ok=True)

    # ---- memory_vs_batch: working set against batch, one panel per design
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=200, sharey=True)
    for ax, design in zip(axes, ("HE-LRM", "IVE")):
        for d, cfg in CONFIGS:
            if d != design:
                continue
            rows, paged = results[(d, cfg)]
            xs, ys = [r["batch"] for r in rows], [mem(r) for r in rows]
            if paged:
                xs.append(paged["batch"])
                ys.append(mem(paged))
            ax.plot(xs, ys, "o-", ms=4, label=label(d, cfg))
        for name, gib in caps:
            ax.axhline(gib, color="black", lw=0.7, ls=":")
            if ax is axes[1]:
                cap_label(ax, name, gib, 1.5)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Batch size")
        ax.set_title(design, fontsize=9)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=5.5, frameon=False, loc="lower right" if design == "HE-LRM" else "upper left")
    axes[0].set_ylabel("Working set (GiB)")
    axes[0].set_ylim(0, 175)
    fig.tight_layout()
    fig.savefig(join(FIG, "memory_vs_batch.png"))
    plt.close(fig)

    # ---- memory_vs_batch_one_panel: the same on one panel, blues against oranges
    fig, ax = plt.subplots(figsize=(4.8, 3.4), dpi=200)
    blues, oranges = plt.colormaps["Blues"], plt.colormaps["Oranges"]
    hl = [c for c in CONFIGS if c[0] == "HE-LRM"]
    iv = [c for c in CONFIGS if c[0] == "IVE"]
    for (d, cfg), i in [(c, i) for i, c in enumerate(hl)] + [(c, i) for i, c in enumerate(iv)]:
        rows, paged = results[(d, cfg)]
        xs, ys = [r["batch"] for r in rows], [mem(r) for r in rows]
        if paged:
            xs.append(paged["batch"])
            ys.append(mem(paged))
        if d == "HE-LRM":
            ax.plot(xs, ys, "-", marker="o", ms=3.5, lw=1.2, color=blues(0.45 + 0.55 * i / (len(hl) - 1)),
                    label=label(d, cfg))
        else:
            ax.plot(xs, ys, "--", marker="s", ms=3.2, lw=1.0, color=oranges(0.4 + 0.6 * i / (len(iv) - 1)),
                    label=label(d, cfg))
    for name, gib in caps:
        ax.axhline(gib, color="black", lw=0.7, ls=":")
        cap_label(ax, name, gib, 1.2)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Working set (GiB)")
    ax.set_ylim(0, 180)
    ax.grid(alpha=0.3, which="both")
    h, l = ax.get_legend_handles_labels()
    leg1 = ax.legend(h[:len(hl)], l[:len(hl)], title="HE-LRM", fontsize=5.5, title_fontsize=6.5, frameon=False,
                     loc="upper left")
    ax.add_artist(leg1)
    ax.legend(h[len(hl):], l[len(hl):], title="IVE", fontsize=5.5, title_fontsize=6.5, frameon=False,
              loc="upper left", bbox_to_anchor=(0.27, 1.0))
    fig.tight_layout()
    fig.savefig(join(FIG, "memory_vs_batch_one_panel.png"))
    plt.close(fig)

    # ---- peak_qps_vs_compression: one line per card and design, from the card's own cells
    fig, ax = plt.subplots(figsize=(4.6, 3.2), dpi=200)
    for (folder, _, cap), color in zip(cards, CARD_PAL):
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
                    miss_marker(ax, x, MISS_Y, design, color)
            ax.plot(xs, ys, ls, marker=marker, ms=3.5, color=color, lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.3, 120)
    ax.set_xlabel("Embedding compression ratio")
    ax.set_ylabel("Peak QPS (largest batch that fits)")
    ax.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([], [], color=c, lw=2, label=n) for (n, _), c in zip(caps, CARD_PAL)]
    handles += [Line2D([], [], color="gray", ls="-", marker="o", ms=3.5, label="HE-LRM"),
                Line2D([], [], color="gray", ls="--", marker="s", ms=3.5, label="IVE"),
                Line2D([], [], color="gray", ls="", marker="x", ms=5, mew=1.2, label="IVE: batch 1 does not fit"),
                (Line2D([], [], color="gray", ls="", marker="x", ms=5, mew=1.2),
                 Line2D([], [], mec="gray", mfc="none", ls="", marker="o", ms=9, mew=0.9))]
    labels = [h.get_label() if not isinstance(h, tuple) else "HE-LRM: batch 1 does not fit" for h in handles]
    ax.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=1, pad=0)}, fontsize=5.5, frameon=False,
              loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(join(FIG, "peak_qps_vs_compression.png"))
    plt.close(fig)

    # ---- peak_qps_bars: cards on x, HE-LRM against IVE at matched compression
    import numpy as np
    fig, axes = plt.subplots(1, len(PAIRS), figsize=(2.6 * len(PAIRS), 2.5), dpi=200, sharey=True)
    ticks = [f"{round(cap)}\n{SHORT[folder]}" for folder, _, cap in cards]
    xs = np.arange(len(cards))
    w = 0.38
    top = 0
    for ax, (title, a, b) in zip(axes, PAIRS):
        for off, key, color in ((-w / 2, a, HELRM_C), (w / 2, b, IVE_C)):
            vals, bs = [], []
            for folder, _, _ in cards:
                r = best(folder, *key)
                vals.append(r["qps"] if r else 0)
                bs.append(r["batch"] if r else None)
            top = max(top, max(vals))
            ax.bar(xs + off, vals, w, color=color, label=f"{key[0]} ({label(*key)})")
            for x, v, bb in zip(xs + off, vals, bs):
                if bb:
                    ax.text(x, v + 1, f"b{bb}", fontsize=5, ha="center", va="bottom", color=color)
                else:
                    miss_marker(ax, x, 1.2, key[0], color, ms=4)
        ax.set_xticks(xs)
        ax.set_xticklabels(ticks, fontsize=6)
        ax.set_title(f"compression {title}", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=5, frameon=False, loc="upper left")
        ax.tick_params(axis="y", labelsize=6)
    axes[0].set_ylabel("Peak QPS (largest batch that fits)", fontsize=7)
    axes[0].set_ylim(0, top * 1.2)
    fig.supxlabel("GPU memory (GiB)", fontsize=7, y=0.02)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.26)
    fig.savefig(join(FIG, "peak_qps_bars.png"))
    plt.close(fig)
    print("wrote " + ", ".join(join(FIG, f) for f in (
        "memory_vs_batch.png", "memory_vs_batch_one_panel.png", "peak_qps_vs_compression.png", "peak_qps_bars.png")))


if __name__ == "__main__":
    main()
