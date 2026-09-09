#!/usr/bin/env python
"""Batch ladder: forward latency, throughput and peak memory against batch
size for HE-LRM (five thresholds) and IVE (six compressions), the largest
batch that fits in device memory, and the first batch past it run with
paging. Figures B (latency against batch) and C (peak QPS against
compression) come from this run.

    python ladder.py [--results DIR]

For each configuration, batches 1, 2, 4, ... climb under
ORION_CHEDDAR_MANAGED_MEMORY=0 with the full protocol (1 warmup + 3 timed
forwards) until a cell dies with CUDA out of memory; that batch is then
run once more with MANAGED_MEMORY=1 (CUDA unified memory, so the excess
pages to host RAM). Cells whose log exists are not re-run, so the script
resumes. Writes the "ladder" section of PAPER_NUMBERS.md and
figures/latency_vs_batch.png, figures/peak_qps_vs_compression.png.

HE-LRM: Cheddar word32, sliced packing, seeded upload,
uploaded at the level compile assigns the sparse layer at that batch.
IVE: one-pass PC-MM, Algorithm 1 VecGen, no bootstrap. The query is
resident, so QPS = batch / forward.
"""
import argparse
import json
import os
import re
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
BATCHES = [1, 2, 4, 8, 16, 32]
CARD_GIB = 97887 / 1024  # RTX PRO 6000, as reported by the driver
OOM_RE = re.compile(r"out of memory|OUT_OF_MEMORY|out_of_memory|cudaErrorMemoryAllocation|"
                    r"bad_alloc|MemoryError", re.I)
CONFIGS = [("HE-LRM", t) for t in THRESHOLDS] + [("IVE", p) for p in IVE_P]


def tag(design, cfg, batch, mm):
    return f"{'t' if design == 'HE-LRM' else 'p'}{cfg}_b{batch}" + ("_mm" if mm else "")


def run_cell(design, cfg, batch, mm, out_dir):
    """Run one cell unless its log exists. Returns (status, json_dict or None):
    status is 'ok', 'oom' or 'failed'."""
    name = tag(design, cfg, batch, mm)
    log = join(out_dir, name + ".log")
    env = {**COMMON_ENV, **GPU_ENV, "ORION_CHEDDAR_MANAGED_MEMORY": "1" if mm else "0"}
    if design == "HE-LRM":
        cell_dir = join(out_dir, name)
        js = join(cell_dir, f"threshold_{cfg}" + ("" if batch == 1 else f"_b{batch}") + ".json")
        cmd = [PY, join(HERE, "bench_sweep.py"), "--compress-threshold", str(cfg), "--base", "4",
               "--word32", "--seeded-upload", "--batch", str(batch), "--upload-level", "min",
               "--out-dir", cell_dir]
    else:
        js = join(out_dir, name + ".json")
        cmd = [PY, join(HERE, "bench_ive_dlrm.py"), "--p", str(cfg), "--ell", str(IVE_ELL[cfg]),
               "--batch", str(batch), "--compress-threshold", "500", "--trials", "3", "--no-ref",
               "--out", js]
    if not exists(log):
        print(f"[{name}]", flush=True)
        run(cmd, env, log)
    if exists(js):
        return "ok", json.load(open(js))
    return ("oom" if OOM_RE.search(open(log).read()) else "failed"), None


def row_from(design, batch, mm, d):
    if design == "HE-LRM":
        fwd, trials = d["runtime_seconds"], d["runtime_seconds_trials"]
        prec, level, cts = d.get("precision"), d.get("upload_level"), d.get("upload_ciphertexts")
    else:
        fwd, prec, level, cts = d["total"], None, d.get("q_level"), 1
        trials = (d.get("trials") or {}).get("total")  # per-trial totals, newer runs only
    peak = d.get("peak_device_mb")
    live = d.get("peak_live_mb")
    keys = d.get("key_memory_mb")
    return {"batch": batch, "mm": mm, "fwd": fwd, "trials": trials, "qps": batch / fwd,
            "peak": peak / 1024 if peak else None, "live": live / 1024 if live else None,
            "keys": keys / 1024 if keys else None, "prec": prec, "level": level, "cts": cts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=expanduser("~/cheddar-orion-results/ladder"))
    args = ap.parse_args()
    os.makedirs(args.results, exist_ok=True)
    from bench_sweep import SPARSE_SIZES
    from dlrm import expanded_width
    total = sum(SPARSE_SIZES)

    def compression(design, cfg):
        if design == "HE-LRM":
            return total / sum(expanded_width(s, cfg, 4) for s in SPARSE_SIZES)
        return IVE_COMPRESSION[cfg]

    problems, results = [], {}  # (design, cfg) -> {"rows": [...], "ceiling": b, "paged": row|None}
    for design, cfg in CONFIGS:
        rows, ceiling, oom_batch = [], None, None
        for b in BATCHES:
            status, d = run_cell(design, cfg, b, False, args.results)
            if status == "ok":
                rows.append(row_from(design, b, False, d))
                ceiling = b
                if design == "HE-LRM" and not d["seeded_upload"]:
                    problems.append(f"{design} {cfg} b{b}: not the seeded config")
                continue
            if status == "oom":
                oom_batch = b
            else:
                problems.append(f"{design} {cfg} b{b}: failed for a reason other than OOM, see "
                                f"{join(args.results, tag(design, cfg, b, False) + '.log')}")
            break
        paged = None
        if oom_batch is not None:
            status, d = run_cell(design, cfg, oom_batch, True, args.results)
            if status == "ok":
                paged = row_from(design, oom_batch, True, d)
                if paged["live"] is None:
                    problems.append(f"{design} {cfg} b{oom_batch} paged: no working-set counter in JSON")
                elif paged["live"] <= CARD_GIB:
                    problems.append(f"{design} {cfg} b{oom_batch}: OOM under managed memory off but "
                                    f"working set {paged['live']:.1f} GiB fits the {CARD_GIB:.1f} GiB "
                                    f"card; the allocator's footprint ({paged['peak']:.1f} GiB) OOMed, "
                                    f"not the computation")
            else:
                problems.append(f"{design} {cfg} b{oom_batch} paged: {status}")
        elif ceiling == BATCHES[-1]:
            problems.append(f"{design} {cfg}: batch {ceiling} still fits; ladder capped")
        results[(design, cfg)] = {"rows": rows, "ceiling": ceiling, "oom": oom_batch, "paged": paged}

    git = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()

    def label(design, cfg):
        return f"t={cfg:,}" if design == "HE-LRM" else f"p={cfg}"

    def gib(x):
        return f"{x:.1f}" if x is not None else "--"

    lines = [
        "## Batch ladder: latency, throughput and memory against batch size (Figures B and C)", "",
        f"Regenerated {date.today().isoformat()} at commit {git} by `ladder.py`. Cheddar word32, "
        "sliced packing, 1 warmup + 3 timed forwards (`fwd` is the mean, `trials` the three "
        "values). HE-LRM: seeded upload, "
        "uploaded at the level compile assigns the sparse layer at that batch. IVE: one-pass PC-MM, "
        "Algorithm 1 VecGen, no bootstrap. The query is resident, so QPS = batch / fwd. "
        "`working set` is the peak of live device bytes (keys, plaintexts, ciphertexts, "
        "temporaries) from a statistics adaptor above Cheddar's pool; `footprint` is what the "
        "pool took from the device; `keys` the resident evaluation keys. Cheddar's pool runs "
        "without bins (`ORION_CHEDDAR_BIN_CHUNK=0`, the default), so footprint equals working set "
        "in every cell; all cells were measured in one pass with this setting. "
        "The card holds 95.6 GiB (drawn and quoted as 96). Each ladder "
        "climbs with managed memory off until CUDA out of memory; the `paged` row is that batch "
        "run with managed memory on (CUDA unified memory), and a check flags any OOM whose paged "
        "working set is under the card.", "",
        "### Every cell", "",
        "| design | config | compression | batch | paged | fwd (s) | trials (s) | QPS | working set (GiB) | "
        "footprint (GiB) | keys (GiB) | selector cts | level | precision (bits) |",
        "|---|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for design, cfg in CONFIGS:
        r = results[(design, cfg)]
        for row in r["rows"] + ([r["paged"]] if r["paged"] else []):
            tr = " / ".join(f"{x:.4f}" for x in row["trials"]) if row["trials"] else "--"
            lines.append(
                f"| {design} | {label(design, cfg)} | {fmt_compression(compression(design, cfg))}x | "
                f"{row['batch']} | {'yes' if row['mm'] else ''} | {row['fwd']:.4f} | {tr} | "
                f"{row['qps']:.2f} | {gib(row['live'])} | {gib(row['peak'])} | "
                f"{row['keys']:.1f} | {row['cts']} | {row['level']} | "
                f"{row['prec']:.2f} |" if row["prec"] is not None else
                f"| {design} | {label(design, cfg)} | {fmt_compression(compression(design, cfg))}x | "
                f"{row['batch']} | {'yes' if row['mm'] else ''} | {row['fwd']:.4f} | {tr} | "
                f"{row['qps']:.2f} | {gib(row['live'])} | {gib(row['peak'])} | {row['keys']:.1f} | "
                f"{row['cts']} | {row['level']} | -- |")
    lines += ["", "### Largest batch that fits, and peak throughput (Figure C)", "",
              "| design | config | compression | largest batch | fwd at it (s) | QPS at it | "
              "working set at it (GiB) | first OOM batch | paged fwd (s) | paged trials (s) | paged QPS | "
              "paged working set (GiB) | best batch | best QPS |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    for design, cfg in CONFIGS:
        r = results[(design, cfg)]
        if not r["rows"]:
            continue
        top = r["rows"][-1]
        best = max(r["rows"], key=lambda x: x["qps"])
        pg = r["paged"]
        pg_tr = (" / ".join(f"{x:.3f}" for x in pg["trials"]) if pg and pg["trials"] else "--")
        lines.append(
            f"| {design} | {label(design, cfg)} | {fmt_compression(compression(design, cfg))}x | "
            f"{top['batch']} | {top['fwd']:.4f} | {top['qps']:.2f} | {gib(top['live'])} | "
            f"{r['oom'] if r['oom'] else '--'} | "
            + (f"{pg['fwd']:.4f} | {pg_tr} | {pg['qps']:.2f} | {gib(pg['live'])} | " if pg
               else "-- | -- | -- | -- | ")
            + f"{best['batch']} | {best['qps']:.2f} |")

    # ---- figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(join(HERE, "figures"), exist_ok=True)
        FIGS = {
            "latency_vs_batch": ("fwd", "Forward latency (s)", True,
                                 "Hollow marker: first batch past the memory ceiling, run with paging"),
            "memory_vs_batch": ("peak", "Peak GPU memory (GiB)", False,
                                "Hollow marker: first batch past the ceiling, run with paging"),
        }

        def panel(ax, design, key, logy, ylabel):
            for design_, cfg in CONFIGS:
                if design_ != design:
                    continue
                r = results[(design, cfg)]
                xs = [row["batch"] for row in r["rows"]]
                ys = [row[key] for row in r["rows"]]  # peak = footprint: the only metric every row has
                line, = ax.plot(xs, ys, "o-", ms=4, label=label(design, cfg))
                if r["paged"]:
                    pv = r["paged"][key]
                    ax.plot([xs[-1], r["paged"]["batch"]], [ys[-1], pv], "--",
                            color=line.get_color(), lw=0.8)
                    ax.plot([r["paged"]["batch"]], [pv], "o", ms=5, mfc="white",
                            color=line.get_color())
            if key == "peak":
                ax.axhline(CARD_GIB, color="black", lw=0.8, ls=":")  # drawn at the real 95.6, labeled 96
                ax.text(BATCHES[0], CARD_GIB + 1.5, "GPU memory, 96 GiB", fontsize=6,
                        va="bottom", ha="left")
            ax.set_xscale("log", base=2)
            if logy:
                ax.set_yscale("log")
            ax.set_xlabel("Batch size")
            if ylabel:
                ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3, which="both")
            ax.legend(fontsize=6, frameon=False, loc="lower right" if key == "peak" else "upper left")

        # Figures A (memory) and B (latency): the two-panel PNG, and each panel
        # on its own for subfigure (a)/(b) with the same y-limits.
        for name, (key, ylabel, logy, note) in FIGS.items():
            fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), dpi=200, sharey=True)
            for ax, design in zip(axes, ("HE-LRM", "IVE")):
                panel(ax, design, key, logy, ylabel if ax is axes[0] else None)
                ax.set_title(design, fontsize=9)
            ylim = axes[0].get_ylim()
            fig.tight_layout()  # the hollow-marker meaning goes in the caption, not on the plot
            fig.savefig(join(HERE, "figures", f"{name}.png"))
            plt.close(fig)
            for design, suffix in (("HE-LRM", "helrm"), ("IVE", "ive")):
                fig, ax = plt.subplots(figsize=(3.6, 2.9), dpi=200)
                panel(ax, design, key, logy, ylabel)
                ax.set_ylim(ylim)
                fig.tight_layout()
                fig.savefig(join(HERE, "figures", f"{name}_{suffix}.png"))
                plt.close(fig)

        fig, ax = plt.subplots(figsize=(4.2, 2.9), dpi=200)
        for design, color, marker in (("HE-LRM", "#2a78d6", "o"), ("IVE", "#eb6834", "s")):
            pts = [(compression(design, cfg), max(results[(design, cfg)]["rows"], key=lambda x: x["qps"]))
                   for d_, cfg in CONFIGS if d_ == design and results[(design, cfg)]["rows"]]
            pts.sort()
            ax.plot([x for x, _ in pts], [b["qps"] for _, b in pts], marker + "-", color=color,
                    label=design + (" (this work)" if design == "HE-LRM" else ""))
            for (x, b), (d_, cfg) in zip(pts, [(d_, cfg) for d_, cfg in sorted(
                    [(d_, cfg) for d_, cfg in CONFIGS if d_ == design and results[(d_, cfg)]["rows"]],
                    key=lambda dc: compression(*dc))]):
                # HE-LRM labels sit below-left of their point, IVE above-right,
                # so the two curves' labels never share a spot where they cross
                # placed by hand for the log-log axes; below-right = (4, -9)/"left" etc.
                place = {("HE-LRM", 5000000): ((4, -9), "left"), ("HE-LRM", 500000): ((-4, 4), "right"),
                         ("HE-LRM", 50000): ((-4, 4), "right"), ("HE-LRM", 5000): ((-4, -9), "right"),
                         ("HE-LRM", 500): ((4, -9), "left"),
                         ("IVE", 1024): ((4, -9), "left"), ("IVE", 512): ((4, -9), "left"),
                         ("IVE", 256): ((-4, 5), "right"), ("IVE", 128): ((4, -9), "left"),
                         ("IVE", 64): ((4, 4), "left"), ("IVE", 32): ((4, 5), "left")}
                off, ha = place[(design, cfg)]
                ax.annotate(f"{label(d_, cfg)}, b={b['batch']}", (x, b["qps"]),
                            textcoords="offset points", xytext=off, ha=ha, fontsize=5)
            # the first batch past the ceiling, run with paging: its QPS as a cross
            paged = [(compression(design, cfg), results[(design, cfg)]["paged"])
                     for d_, cfg in CONFIGS if d_ == design and results[(design, cfg)]["paged"]]
            if paged:
                ax.plot([x for x, _ in paged], [r["qps"] for _, r in paged], "x", color=color,
                        ms=5, mew=1.2, ls="none")
                for (x, r), (d_, cfg) in zip(paged, [(d_, cfg) for d_, cfg in CONFIGS
                                                      if d_ == design and results[(d_, cfg)]["paged"]]):
                    if design == "HE-LRM":
                        off, ha = (0, 5), "center"
                    else:  # the three IVE b=16 crosses sit close: spread their labels
                        off, ha = {256: ((-5, -9), "right"), 128: ((5, -9), "left"), 64: ((5, -9), "left")}.get(cfg, ((0, -9), "center"))
                    ax.annotate(f"b={r['batch']}", (x, r["qps"]), textcoords="offset points",
                                xytext=off, ha=ha, fontsize=4.5, color=color)
        from matplotlib.lines import Line2D
        handles, labels_ = ax.get_legend_handles_labels()
        handles.append(Line2D([], [], marker="x", color="gray", ls="none", ms=5, mew=1.2))
        labels_.append("first batch past the ceiling, paged")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(7, 1.5e5)
        ax.set_ylim(0.3, 120)
        ax.set_xlabel("Embedding compression ratio")
        ax.set_ylabel("Peak throughput (queries/s)")
        ax.legend(handles, labels_, loc="upper left", fontsize=6, frameon=False)
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(join(HERE, "figures", "peak_qps_vs_compression.png"))
        lines += ["", "Figures: `figures/memory_vs_batch.png` (A), `figures/latency_vs_batch.png` (B), "
                  "`figures/peak_qps_vs_compression.png` (C); A and B also as one panel per design, "
                  "`*_helrm.png` (a) and `*_ive.png` (b), same y-limits as the pair."]
    except Exception as exc:
        problems.append(f"figures not written: {exc}")

    lines += ["", "Checks: " + ("all passed." if not problems else " ".join(problems))]
    write_section("ladder", lines)


if __name__ == "__main__":
    main()
