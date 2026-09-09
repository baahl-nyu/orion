#!/usr/bin/env python
"""Working set against batch with the GPUs' capacities, and peak throughput per GPU.

    python gpu_ceilings.py [--results DIR]

Three figures under figures/, from the cells under --results (default
~/cheddar-orion-results/final), all Cheddar word32, bins off, managed memory
off, 1 warmup + 3 timed forwards:

  gpu_ceilings_memory_vs_batch.png   working set of a batch against batch size,
                                     one panel per design, from the RTX PRO 6000
                                     climb (`ladder/`, ladder.py) including the
                                     first paged batch; the capacity of each
                                     GPU we ran on is a horizontal line. The
                                     working set does not depend on the card
                                     (max_qps.py checks this), so one climb
                                     gives every card's ceiling.
  gpu_ceilings_memory_one_panel.png  the same on one panel.
  gpu_ceilings_peak_qps.png          peak throughput against compression, one
                                     panel per GPU, from that GPU's own cells:
                                     the largest batch that ran there without
                                     OOM, divided by its forward on that card.
                                     A cross marks a configuration whose batch
                                     1 does not fit (IVE); HE-LRM's would be a
                                     cross in a circle.

The numbers behind the third figure are tabulated in max_qps.md; behind the
first two, in PAPER_NUMBERS.md (batch ladder).
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

from ladder import CONFIGS, IVE_COMPRESSION, row_from, tag  # noqa: E402
from max_qps import CARDS, load_cells  # noqa: E402
from table2 import fmt_compression  # noqa: E402

FIG = join(HERE, "figures")
BATCHES = (1, 2, 4, 8, 16, 32)
# short card names for the capacity lines and panel titles, keyed by folder
SHORT = {"ladder": "RTX PRO 6000", "ladder_a100": "A100", "ladder_3090": "RTX 3090", "ladder_h100": "H100"}
HELRM_C, IVE_C = "#2a78d6", "#eb6834"


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
    os.makedirs(FIG, exist_ok=True)

    # ---- 1: working set against batch, one panel per design
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
    fig.savefig(join(FIG, "gpu_ceilings_memory_vs_batch.png"))
    plt.close(fig)

    # ---- 2: the same on one panel, blues against oranges
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
    fig.savefig(join(FIG, "gpu_ceilings_memory_one_panel.png"))
    plt.close(fig)

    # ---- 3: peak throughput against compression, one panel per GPU, each from its own cells
    fig, axes = plt.subplots(1, len(cards), figsize=(1.9 * len(cards), 2.4), dpi=200, sharey=True, sharex=True)
    for ax, (folder, name, cap) in zip(axes, cards):
        cells, _ = load_cells(join(args.results, folder))
        for design, color, marker in (("HE-LRM", HELRM_C, "o"), ("IVE", IVE_C, "s")):
            xs, ys, miss = [], [], []
            for d, cfg in CONFIGS:
                if d != design:
                    continue
                fit = [r for (dd, c, b), r in cells.items() if dd == d and c == cfg]
                x = compression(d, cfg)
                if fit:
                    best = max(fit, key=lambda r: r["batch"])
                    xs.append(x)
                    ys.append(best["qps"])
                    ax.annotate(f"b{best['batch']}", (x, best["qps"]), (0, 4 if design == "HE-LRM" else -8),
                                textcoords="offset points", fontsize=4.5, ha="center", color=color)
                else:
                    miss.append(x)
            ax.plot(xs, ys, "-", marker=marker, ms=3, color=color, lw=1, label=design)
            if miss:  # batch 1 does not fit: IVE a cross, HE-LRM a cross in a circle
                ax.plot(miss, [0.45] * len(miss), "x", color=color, ms=4, mew=1.1)
                if design == "HE-LRM":
                    ax.plot(miss, [0.45] * len(miss), "o", mfc="none", mec=color, ms=7, mew=0.9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.35, 100)
        ax.set_title(f"{round(cap)} GiB\n({SHORT[folder]})", fontsize=7)
        ax.grid(alpha=0.3, which="both")
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel("Peak QPS", fontsize=7)
    axes[0].legend(fontsize=6, frameon=False, loc="upper left")
    axes[0].text(12, 0.6, "batch 1 does not fit:  x IVE,  ⊗ HE-LRM", fontsize=4.5, va="bottom", color="gray")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24)
    fig.supxlabel("Embedding compression ratio", fontsize=7, y=0.04)
    fig.savefig(join(FIG, "gpu_ceilings_peak_qps.png"))
    plt.close(fig)
    print("wrote", ", ".join(join(FIG, f) for f in ("gpu_ceilings_memory_vs_batch.png", "gpu_ceilings_memory_one_panel.png",
                                                     "gpu_ceilings_peak_qps.png")))


if __name__ == "__main__":
    main()
