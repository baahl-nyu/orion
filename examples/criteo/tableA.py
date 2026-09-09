#!/usr/bin/env python
"""Table A: selector cost by stage at batch 1, HE-LRM against IVE, and the
selector-time-against-compression figure.

    python tableA.py [--results DIR] [--batch1 DIR]

HE-LRM stage times (BSGS lookup, repack to the top-MLP layout) are read
from the batch-1 run's GPU JSONs (batch1.py, --stage-times), so run that
first. IVE cells (p = 32 .. 1024, one-pass PC-MM, Algorithm 1 VecGen, no
bootstrap, 1 warmup + 3 timed forwards) are run here if missing. Writes
the "tableA" section of PAPER_NUMBERS.md and
figures/selector_cost_vs_compression.png.
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import date
from os.path import abspath, dirname, exists, expanduser, join

HERE = dirname(abspath(__file__))
REPO = dirname(dirname(HERE))
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from table2 import COMMON_ENV, GPU_ENV, PY, THRESHOLDS, fmt_compression, run, write_section

IVE_P = [32, 64, 128, 256, 512, 1024]
IVE_ELL = {32: 5, 64: 4, 128: 4, 256: 3, 512: 3, 1024: 3}
IVE_COMPRESSION = {32: 14453, 64: 8374, 128: 4628, 256: 2536, 512: 1346, 1024: 717}
MS = 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=expanduser("~/cheddar-orion-results/tableA"))
    ap.add_argument("--batch1", default=expanduser("~/cheddar-orion-results/batch1"))
    args = ap.parse_args()
    os.makedirs(args.results, exist_ok=True)
    from bench_sweep import SPARSE_SIZES
    from dlrm import expanded_width
    total = sum(SPARSE_SIZES)
    problems = []

    ours = {}
    for t in THRESHOLDS:
        js = join(args.batch1, "gpu", f"threshold_{t}.json")
        if not exists(js):
            problems.append(f"HE-LRM t={t}: no batch-1 GPU JSON at {js}; run batch1.py first")
            continue
        d = json.load(open(js))
        st = d.get("stage_seconds")
        if not st:
            problems.append(f"HE-LRM t={t}: batch-1 JSON has no stage times")
            continue
        width = sum(expanded_width(s, t, 4) for s in SPARSE_SIZES)
        ours[t] = {"compression": total / width, "lookup": st["lookup"] * MS,
                   "repack": st["sparse_to_top"] * MS, "forward": d["runtime_seconds"] * MS,
                   "bottom": st["bottom_mlp"] * MS, "top": st["top_mlp"] * MS}

    ive = {}
    for p in IVE_P:
        js, log = join(args.results, f"p{p}_b1.json"), join(args.results, f"p{p}_b1.log")
        if not exists(js):
            print(f"[ive] p={p}", flush=True)
            run([PY, join(HERE, "bench_ive_dlrm.py"), "--p", str(p), "--ell", str(IVE_ELL[p]),
                 "--batch", "1", "--compress-threshold", "500", "--trials", "3", "--no-ref",
                 "--out", js], {**COMMON_ENV, **GPU_ENV}, log)
        if not exists(js):
            problems.append(f"IVE p={p}: run failed, see {log}")
            continue
        d = json.load(open(js))
        if d.get("pcmm_kind") != "matmul" or d.get("boot_protocol"):
            problems.append(f"IVE p={p}: not one-pass PC-MM without bootstrap")
        ive[p] = {"compression": IVE_COMPRESSION[p], "vecgen": d["vecgen"] * MS,
                  "pcmm": d["pcmm"] * MS, "sum": d["sum"] * MS, "repack": d["repack"] * MS,
                  "boot": d["boot"] * MS, "forward": d["total"] * MS,
                  "bottom": d["bottom"] * MS, "top": d["mlp"] * MS}
        ive[p]["selector"] = ive[p]["vecgen"] + ive[p]["pcmm"] + ive[p]["sum"] + ive[p]["repack"]
    for t in ours:
        ours[t]["selector"] = ours[t]["lookup"] + ours[t]["repack"]

    git = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    cols = [("HE-LRM", f"t={t:,}", ours.get(t)) for t in THRESHOLDS] + \
           [("IVE", f"p={p}", ive.get(p)) for p in IVE_P]
    stage_rows = [
        ("Power tree (VecGen)", "vecgen"), ("PC-MM", "pcmm"), ("Sub-table sum", "sum"),
        ("BSGS lookup", "lookup"), ("Repack to top-MLP layout", "repack"), ("Selector total", "selector"),
    ]
    ctx_rows = [("Bottom MLP", "bottom"), ("Add + top MLP", "top"), ("Forward total", "forward")]

    def cell(d, key, fmt="%.1f"):
        if d is None:
            return "—"
        if key not in d:
            return "0"
        return fmt % d[key]

    lines = [
        "## Table A: selector cost by stage at batch 1, HE-LRM against IVE (ms)", "",
        f"Regenerated {date.today().isoformat()} at commit {git} by `tableA.py`. Cheddar word32, "
        "sliced packing, managed memory off, 1 warmup + 3 timed forwards, stage means. HE-LRM: "
        "seeded upload at the sparse layer's level; its stages come from the "
        "batch-1 run (batch1.py). IVE: one-pass PC-MM, Algorithm 1 VecGen, no bootstrap, 432-slot "
        "query. A stage a design does not have is 0.", "",
        "| Stage | " + " | ".join(f"{d} {c}" for d, c, _ in cols) + " |",
        "|---|" + "---:|" * len(cols),
        "| Compression | " + " | ".join(
            (fmt_compression(x["compression"]) if x else "—") + "x" for _, _, x in cols) + " |",
    ]
    for label, key in stage_rows:
        lines.append(f"| {label} | " + " | ".join(cell(x, key) for _, _, x in cols) + " |")
    lines += ["", "Context, same forwards (ms):", "",
              "| | " + " | ".join(f"{d} {c}" for d, c, _ in cols) + " |", "|---|" + "---:|" * len(cols)]
    for label, key in ctx_rows:
        lines.append(f"| {label} | " + " | ".join(cell(x, key) for _, _, x in cols) + " |")

    # figure
    fig_path = join(HERE, "figures", "selector_cost_vs_compression.png")
    try:
        import matplotlib.pyplot as plt
        from figstyle import COL_W, apply, save, series
        apply()
        fig, ax = plt.subplots(figsize=(COL_W, 2.7))
        ts = [t for t in THRESHOLDS if t in ours]
        xs, ys = [ours[t]["compression"] for t in ts], [ours[t]["selector"] for t in ts]
        series(ax, xs, ys, 1, label="HE-LRM (this work)", marker="o")
        for t, x, y in zip(ts, xs, ys):
            ax.annotate(f"t={t:,}", (x, y), textcoords="offset points",
                        xytext=(7, 4) if x < 100 else (0, -12), fontsize=6, ha="center" if x >= 100 else "left")
        ps = [p for p in IVE_P if p in ive]
        xs, ys = [ive[p]["compression"] for p in ps], [ive[p]["selector"] for p in ps]
        series(ax, xs, ys, 0, label="IVE", marker="o")
        for p, x, y in zip(ps, xs, ys):
            ax.annotate(f"p={p}", (x, y), textcoords="offset points",
                        xytext=(0, -12) if p == 32 else (7, 4), fontsize=6, ha="center" if p == 32 else "left")
        ax.set_xscale("log")
        ax.set_ylim(-18, 185)
        ax.set_yticks(range(0, 176, 25))
        ax.set_xlabel("Embedding compression ratio")
        ax.set_ylabel("Selector time (ms)")
        ax.legend(loc="upper left")
        fig.tight_layout()
        save(fig, "selector_cost_vs_compression")
        plt.close(fig)
        lines += ["", f"Figure: `figures/selector_cost_vs_compression.png` (selector total against "
                  "compression, both designs, batch 1)."]
    except Exception as exc:  # the table is the deliverable; the figure is best effort
        problems.append(f"figure not written: {exc}")

    lines += ["", "Checks: " + ("all passed." if not problems else " ".join(problems))]
    write_section("tableA", lines)


if __name__ == "__main__":
    main()
