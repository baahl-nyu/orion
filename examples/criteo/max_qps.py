#!/usr/bin/env python
"""Largest batch that fits, and the throughput it buys, per GPU.

    python max_qps.py [--results DIR]

Reads the measured ladder cells of every card under --results (default
examples/criteo/results): `ladder/` is the RTX PRO 6000 climb
(ladder.py), `ladder_<card>/` the cells run on another card with the same
benches, command lines and environment (bins off, managed memory off,
1 warmup + 3 timed forwards). Writes examples/criteo/max_qps.md (whole file).

Nothing here is estimated: a batch counts as fitting on a card only if it
was run there without OOM, and the throughput is batch / that card's own
forward time. The working set of a cell does not depend on the card (the
check below confirms it), so the RTX PRO 6000 climb says which batch is the
next one up and what it would need; the row states whether that next batch
was tried on the card (OOM) or is excluded by its working set.
"""
import argparse
import glob
import json
import os
import re
import sys
from datetime import date
from os.path import abspath, basename, dirname, exists, expanduser, join

HERE = dirname(abspath(__file__))
REPO = dirname(dirname(HERE))
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ladder import CONFIGS, IVE_COMPRESSION, row_from, tag  # noqa: E402
from table2 import THRESHOLDS, fmt_compression  # noqa: E402

OUT_MD = join(HERE, "max_qps.md")

# card folder -> (label, capacity in GiB as the driver reports it)
CARDS = [
    ("ladder", "RTX PRO 6000 (96 GiB)", 97887 / 1024),
    ("ladder_a100", "A100-SXM4 (40 GiB)", 40960 / 1024),
    ("ladder_3090", "RTX 3090 (24 GiB)", 24576 / 1024),
    ("ladder_h100", "H100 (80 GiB)", 81559 / 1024),
]


def load_cells(root):
    """{(design, cfg, batch): row} for the managed-memory-off cells under root,
    plus {(design, cfg, batch): 'oom'} for cells whose log exists without a JSON."""
    cells, ooms = {}, set()
    for design, cfg in CONFIGS:
        for batch in (1, 2, 4, 8, 16, 32):
            name = tag(design, cfg, batch, False)
            if design == "HE-LRM":
                js = glob.glob(join(root, name, "*.json"))
            else:
                js = [join(root, name + ".json")] if exists(join(root, name + ".json")) else []
            if js:
                cells[(design, cfg, batch)] = row_from(design, batch, False, json.load(open(js[0])))
            elif exists(join(root, name + ".log")):
                ooms.add((design, cfg, batch))
    return cells, ooms


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
        return f"t={cfg:,}" if design == "HE-LRM" else f"p={cfg}"

    def gpu_info(root):
        f = join(root, "gpu.txt")
        return open(f).read().strip().splitlines()[0] if exists(f) else None

    ref_cells, ref_ooms = load_cells(join(args.results, "ladder"))
    data = {folder: load_cells(join(args.results, folder)) for folder, _, _ in CARDS}
    problems = []

    # ---- per-card summary rows
    summary = {}  # (folder, design, cfg) -> dict
    for folder, name, cap in CARDS:
        cells, ooms = data[folder]
        for design, cfg in CONFIGS:
            fit = [b for (d, c, b) in cells if d == design and c == cfg]
            if not fit:
                nxt = 1
                row = None
            else:
                best = max(fit)
                row = cells[(design, cfg, best)]
                nxt = best * 2
            need = ref_cells.get((design, cfg, nxt))
            if (design, cfg, nxt) in ooms:
                why = "OOM on this card"
            elif need is not None:
                why = f"needs {need['live']:.1f} GiB"
                if need["live"] <= cap - 1.0 and folder != "ladder":
                    problems.append(f"{name} {label(design, cfg)}: batch {nxt} needs {need['live']:.1f} GiB "
                                    f"< {cap:.1f} GiB but was not run on this card")
            elif (design, cfg, nxt) in ref_ooms:
                why = "OOM on the RTX PRO 6000 too"
            else:
                why = "not run anywhere"
            summary[(folder, design, cfg)] = {"row": row, "best": (max(fit) if fit else 0), "next": nxt, "why": why}

    # ---- checks: working sets agree across cards for the same cell
    for folder, name, cap in CARDS[1:]:
        cells, _ = data[folder]
        for key, row in cells.items():
            ref = ref_cells.get(key)
            if ref and row["live"] and ref["live"] and abs(row["live"] - ref["live"]) > 0.15:
                problems.append(f"{name} {label(key[0], key[1])} b{key[2]}: working set {row['live']:.1f} GiB "
                                f"vs {ref['live']:.1f} GiB on the RTX PRO 6000")

    git = os.popen(f"git -C {REPO} rev-parse --short HEAD").read().strip()
    lines = [
        "# Largest batch that fits, and peak throughput, per GPU", "",
        f"Regenerated {date.today().isoformat()} at commit {git} by `max_qps.py` from "
        f"`{os.path.relpath(args.results, REPO)}/`. Every cell is a measured forward on the card named: "
        "Cheddar word32, bins off (`ORION_CHEDDAR_BIN_CHUNK=0`), managed memory off, 1 warmup + 3 timed "
        "forwards, the same command lines as `ladder.py` (HE-LRM: seeded upload at the sparse layer's level; "
        "IVE: one-pass PC-MM, Algorithm 1 VecGen). The query is resident, so QPS = batch / forward. "
        "`largest batch` is the largest batch that ran on that card without CUDA out of memory; the "
        "`next batch` column says why the next power of two is out: it was tried on the card and OOMed, or "
        "its working set, measured on the RTX PRO 6000 climb and identical across cards (checked below), "
        "exceeds the card. A card's forward times are its own; nothing is scaled from another card.", "",
        "Cards:", "",
    ]
    for folder, name, cap in CARDS:
        info = gpu_info(join(args.results, folder))
        lines.append(f"* `{folder}/`: {name}, {cap:.1f} GiB as reported by the driver"
                     + (f"; `gpu.txt`: {info}" if info else "") + ".")
    lines += ["", "## Per card", ""]
    for folder, name, cap in CARDS:
        lines += [f"### {name}", "",
                  "| design | config | compression | largest batch | fwd (s) | trials (s) | QPS | working set (GiB) | next batch |",
                  "|---|---|---:|---:|---:|---|---:|---:|---|"]
        for design, cfg in CONFIGS:
            s = summary[(folder, design, cfg)]
            r = s["row"]
            if r is None:
                lines.append(f"| {design} | {label(design, cfg)} | {fmt_compression(compression(design, cfg))}x | "
                             f"none | -- | -- | 0 | -- | batch 1: {s['why']} |")
                continue
            trials = " / ".join(f"{x:.3f}" for x in r["trials"]) if r["trials"] else "--"
            lines.append(f"| {design} | {label(design, cfg)} | {fmt_compression(compression(design, cfg))}x | "
                         f"{s['best']} | {r['fwd']:.3f} | {trials} | {r['qps']:.1f} | {r['live']:.1f} | "
                         f"batch {s['next']}: {s['why']} |")
        lines.append("")

    # ---- side by side
    lines += ["## Side by side", "",
              "| design | config | compression | " + " | ".join(f"{n}: batch / QPS" for _, n, _ in CARDS) + " |",
              "|---|---|---:|" + "|".join("---:" for _ in CARDS) + "|"]
    for design, cfg in CONFIGS:
        cells = []
        for folder, name, cap in CARDS:
            s = summary[(folder, design, cfg)]
            cells.append(f"{s['best']} / {s['row']['qps']:.1f}" if s["row"] else "does not fit")
        lines.append(f"| {design} | {label(design, cfg)} | {fmt_compression(compression(design, cfg))}x | "
                     + " | ".join(cells) + " |")

    # ---- latency ratio at batch 1
    lines += ["", "## Forward time against the RTX PRO 6000, batch 1", "",
              "| design | config | " + " | ".join(f"{n} (s)" for _, n, _ in CARDS)
              + " | " + " | ".join(f"{n} / RTX PRO 6000" for _, n, _ in CARDS[1:]) + " |",
              "|---|---|" + "|".join("---:" for _ in CARDS) + "|" + "|".join("---:" for _ in CARDS[1:]) + "|"]
    for design, cfg in CONFIGS:
        fw = [data[f][0].get((design, cfg, 1)) for f, _, _ in CARDS]
        vals = [f"{r['fwd']:.3f}" if r else "OOM" for r in fw]
        ratios = [f"{r['fwd'] / fw[0]['fwd']:.2f}x" if (r and fw[0]) else "--" for r in fw[1:]]
        lines.append(f"| {design} | {label(design, cfg)} | " + " | ".join(vals) + " | " + " | ".join(ratios) + " |")

    lines += ["", "Checks: " + ("; ".join(problems) if problems else "all passed") + ". A cell with a working set within "
              "1 GiB of a card's capacity that was not run on that card would be flagged here; working sets are "
              "compared across cards to 0.15 GiB.", ""]
    open(OUT_MD, "w").write("\n".join(lines))
    print(f"wrote {OUT_MD}")
    if problems:
        print("Checks:", *problems, sep="\n  ")


if __name__ == "__main__":
    main()
