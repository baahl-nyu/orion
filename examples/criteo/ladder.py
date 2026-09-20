#!/usr/bin/env python
"""Batch ladder: forward latency, throughput and peak memory against batch
size for HE-LRM (five thresholds) and IVE (six compressions), the largest
batch that fits in device memory, and the first batch past it run with
paging. Figure B (HE-LRM latency against batch, one column-width panel)
comes from this run; the memory and peak-throughput figures are drawn by
gpu_ceilings.py from these cells and the other cards' folders.

    python ladder.py [--results DIR]

For each configuration, batches 1, 2, 4, ... climb under
ORION_CHEDDAR_MANAGED_MEMORY=0 with the full protocol (1 warmup + 3 timed
forwards) until a cell dies with CUDA out of memory; that batch is then
run once more with MANAGED_MEMORY=1 (CUDA unified memory, so the excess
pages to host RAM). Cells whose log exists are not re-run, so the script
resumes. Writes the "ladder" section of PAPER_NUMBERS.md and
figures/latency_vs_batch.png.

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
    ap.add_argument("--results", default=join(HERE, "results", "ladder"))
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
        "## Batch ladder: latency, throughput and memory against batch size (Figure B)", "",
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
    lines += ["", "### Largest batch that fits, and peak throughput on the RTX PRO 6000", "",
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
        import matplotlib.pyplot as plt
        from figstyle import COL_W, LINE, apply, save, series
        apply()
        # Figure B (latency): HE-LRM at every threshold on one column-width panel; the IVE
        # ladders are in the table above. The legend sits over batches 1-4, under the paged points.
        fig, ax = plt.subplots(figsize=(COL_W, 2.0))
        for i, (design, cfg) in enumerate([c for c in CONFIGS if c[0] == "HE-LRM"]):
            r = results[(design, cfg)]
            xs = [row["batch"] for row in r["rows"]]
            ys = [row["fwd"] for row in r["rows"]]
            series(ax, xs, ys, i, label=f"{label(design, cfg)}, {fmt_compression(compression(design, cfg))}x")
            if r["paged"]:  # first batch past the ceiling, paged: dotted segment, open marker
                pb, pv = r["paged"]["batch"], r["paged"]["fwd"]
                ax.plot([xs[-1], pb], [ys[-1], pv], ":", color=LINE[i], lw=1.0, zorder=2)
                ax.plot([pb], [pv], "o", mfc="white", mec=LINE[i], mew=1.2, ls="none", zorder=3)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_ylim(0.1, 40)
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Forward latency (s)")
        ax.legend(loc="upper left", fontsize=6, borderpad=0.3, labelspacing=0.2, handletextpad=0.3)
        fig.tight_layout()
        save(fig, "latency_vs_batch")
        plt.close(fig)
        lines += ["", "Figure: `figures/latency_vs_batch.png` (B), the HE-LRM ladders only; the IVE ladders "
                  "are in the table above. The memory-against-batch figure with the cards' capacities and the "
                  "peak-throughput figures are drawn by `gpu_ceilings.py` from these cells and the other "
                  "cards' folders (see `max_qps.md`)."]
    except Exception as exc:
        problems.append(f"figures not written: {exc}")

    lines += ["", "Checks: " + ("all passed." if not problems else " ".join(problems))]
    write_section("ladder", lines)


if __name__ == "__main__":
    main()
