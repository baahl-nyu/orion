#!/usr/bin/env python
"""CPU against GPU at batch 1: ciphertexts per query, slot utilization and
forward runtime per threshold, current compiler (sliced diagonal packing).

    python batch1.py [--results DIR]

Runs every missing cell, then writes the "batch1" section of
examples/criteo/PAPER_NUMBERS.md (replaced in place). Raw JSON
and logs go to --results (default ~/cheddar-orion-results/batch1). A cell
whose JSON exists is not re-run; delete it to redo.

CPU: table2.py --cpu-cell, Lattigo, 64-bit chain (params.yaml), single
     thread, 1 warmup + 3 timed forwards.
GPU: bench_sweep.py, Cheddar word32 chain, seeded upload at the sparse
     layer's level, managed memory off, 1 warmup + 3 timed forwards (the
     timed region includes re-expanding the uploaded seeds).
"""
import argparse
import json
import math
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

from table2 import (COMMON_ENV, CPU_ENV, GPU_ENV, PY, SLOTS_PER_CT, THRESHOLDS,
                    fmt_compression, run, sparse_level_from_log, write_section)

TRIALS = 3
GPU_LEVEL = {500: 4, 5000: 4, 50000: 4, 500000: 5, 5000000: 5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=expanduser("~/cheddar-orion-results/batch1"))
    args = ap.parse_args()
    from bench_sweep import SPARSE_SIZES
    from dlrm import dense_features, expanded_width

    gpu_dir, cpu_dir = join(args.results, "gpu"), join(args.results, "cpu")
    os.makedirs(gpu_dir, exist_ok=True)
    os.makedirs(cpu_dir, exist_ok=True)
    total = sum(SPARSE_SIZES)
    notes, gpu, cpu = [], {}, {}

    for t in THRESHOLDS:
        js, log = join(gpu_dir, f"threshold_{t}.json"), join(gpu_dir, f"gpu_t{t}.log")
        if not exists(js):
            print(f"[gpu] t={t}", flush=True)
            run([PY, join(HERE, "bench_sweep.py"), "--compress-threshold", str(t),
                 "--base", "4", "--word32", "--seeded-upload", "--batch", "1",
                 "--upload-level", str(GPU_LEVEL[t]), "--out-dir", gpu_dir, "--stage-times"],
                {**COMMON_ENV, **GPU_ENV}, log)
        if exists(js):
            gpu[t] = json.load(open(js))
            gpu[t]["sparse_level"] = sparse_level_from_log(log)
        else:
            notes.append(f"GPU t={t}: run failed, see {log}")

    for t in THRESHOLDS:
        js, log = join(cpu_dir, f"cpu_t{t}.json"), join(cpu_dir, f"cpu_t{t}.log")
        if not exists(js):
            print(f"[cpu] t={t}", flush=True)
            run([PY, join(HERE, "table2.py"), "--cpu-cell", str(t), "--trials", str(TRIALS),
                 "--out", js], {**COMMON_ENV, **CPU_ENV}, log)
        if exists(js):
            cpu[t] = json.load(open(js))
        else:
            notes.append(f"CPU t={t}: run failed, see {log}")

    for t in THRESHOLDS:
        width = sum(expanded_width(s, t, 4) for s in SPARSE_SIZES)
        if t in cpu and cpu[t]["sparse_width"] != width:
            notes.append(f"CPU t={t}: encrypted width {cpu[t]['sparse_width']} != model width {width}")
        if t in cpu and len(cpu[t]["runtime_seconds_trials"]) != TRIALS:
            notes.append(f"CPU t={t}: {len(cpu[t]['runtime_seconds_trials'])} trials, expected {TRIALS}")
        if t in gpu:
            g = gpu[t]
            if g["upload_ciphertexts"] != math.ceil(width / SLOTS_PER_CT):
                notes.append(f"GPU t={t}: {g['upload_ciphertexts']} uploaded cts, width {width} "
                             f"needs {math.ceil(width / SLOTS_PER_CT)}")
            if g["upload_level"] != g["sparse_level"]:
                notes.append(f"GPU t={t}: upload level {g['upload_level']} != compiled sparse "
                             f"level {g['sparse_level']}")
            if not g["seeded_upload"] or \
                    g["env_knobs"].get("ORION_CHEDDAR_MANAGED_MEMORY") != "0":
                notes.append(f"GPU t={t}: not the seeded / managed-memory-off config")
            if len(g["runtime_seconds_trials"]) != TRIALS:
                notes.append(f"GPU t={t}: {len(g['runtime_seconds_trials'])} trials, expected {TRIALS}")
        if t in cpu and t in gpu and cpu[t]["sparse_level"] != gpu[t]["sparse_level"]:
            notes.append(f"t={t}: sparse layer at level {cpu[t]['sparse_level']} on CPU, "
                         f"{gpu[t]['sparse_level']} on GPU")

    git = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    lines = [
        "## CPU against GPU at batch 1", "",
        f"Regenerated {date.today().isoformat()} at commit {git} by `batch1.py`: current compiler "
        "(sliced diagonal packing), batch 1, 1 warmup + 3 timed forwards; "
        "runtime is the mean of the 3 timed forwards, with the per-trial spread alongside.",
        f"Ciphertexts/query and slot utilization are for the one-hot selector "
        f"({SLOTS_PER_CT:,} slots per ciphertext); the {dense_features} dense features travel "
        "in one further ciphertext at every threshold. CPU: Lattigo, 64-bit chain (params.yaml), "
        "one thread. GPU: Cheddar, word32 chain, seeded upload at the sparse layer's level, "
        "managed memory off; the timed forward includes re-expanding the uploaded seeds.", "",
        "| Threshold | Compression | Ciphertexts/query | Slot utilization | CPU lookup (s) | "
        "CPU runtime (s) | GPU lookup (s) | GPU runtime (s) | Speedup |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    spread = []
    stage_rows = {"bottom_mlp": "Bottom MLP", "lookup": "Embedding lookup",
                  "sparse_to_top": "Repack lookup to top layout", "add": "Add",
                  "top_mlp": "Top MLP"}
    breakdown = ["", "### Forward breakdown by stage (s, mean of the 3 timed forwards)", "",
                 "Timed inside the same forwards as the runtimes above; the remainder is "
                 "Python glue and, on the GPU, re-expanding the uploaded seeds.", "",
                 "| Stage | " + " | ".join(f"CPU {t:,}" for t in THRESHOLDS) + " | "
                 + " | ".join(f"GPU {t:,}" for t in THRESHOLDS) + " |",
                 "|---|" + "---:|" * (2 * len(THRESHOLDS))]
    for key, label in stage_rows.items():
        cells = []
        for d in (cpu, gpu):
            for t in THRESHOLDS:
                st = (d.get(t) or {}).get("stage_seconds") or {}
                cells.append(f"{st[key]:.3f}" if key in st else "—")
        breakdown.append(f"| {label} | " + " | ".join(cells) + " |")
    for t in THRESHOLDS:
        width = sum(expanded_width(s, t, 4) for s in SPARSE_SIZES)
        cts = math.ceil(width / SLOTS_PER_CT)
        util = width / (cts * SLOTS_PER_CT)
        c, g = cpu.get(t), gpu.get(t)
        cr = f"{c['runtime_seconds']:.1f}" if c else "—"
        gr = f"{g['runtime_seconds']:.3f}" if g else "—"
        cl = f"{c['stage_seconds']['lookup']:.1f}" if c and c.get("stage_seconds") else "—"
        gl = f"{g['stage_seconds']['lookup']:.3f}" if g and g.get("stage_seconds") else "—"
        sp = f"{c['runtime_seconds'] / g['runtime_seconds']:,.0f}x" if c and g else "—"
        lines.append(f"| {t:,} | {fmt_compression(total / width)}x | {cts} | {util:.1%} | "
                     f"{cl} | {cr} | {gl} | {gr} | {sp} |")
        if c and g:
            spread.append(f"{t:,}: CPU {min(c['runtime_seconds_trials']):.1f}–"
                          f"{max(c['runtime_seconds_trials']):.1f} s, GPU "
                          f"{min(g['runtime_seconds_trials']):.3f}–{max(g['runtime_seconds_trials']):.3f} s")
    lines += ["", "Per-trial spread (min–max of the 3 timed forwards): " + "; ".join(spread) + "."]
    lines += breakdown
    lines += ["", "Checks: " + ("all passed." if not notes else " ".join(notes)), ""]

    write_section("batch1", lines)


if __name__ == "__main__":
    main()
