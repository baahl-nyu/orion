#!/usr/bin/env python
"""Table 2: compression ratio, slots and peak memory per threshold,
measured on the CPU (Lattigo) and GPU (Cheddar) backends with the current
compiler (sliced diagonal packing), batch 1.

    python table2.py [--results DIR]

Runs every missing cell, then writes examples/criteo/table2.md. Raw JSON
and logs go to --results (default ~/cheddar-orion-results/table2, outside
the repo). A cell whose JSON exists is not re-run; delete it to redo.

GPU: bench_sweep.py, word32 chain, seeded upload at the level the sparse
     layer consumes (checked against the compiled module tree), managed
     memory off so a cell that does not fit in VRAM fails instead of
     paging. Peak = device allocation peak over setup + 1 warmup + 3 forwards.
CPU: this file's --cpu-cell mode, Lattigo, 64-bit chain from params.yaml
     (Lattigo takes LogQ/LogP bit lists only; the word32 chain is
     Cheddar-only), single thread like the paper's CPU column. Peak =
     process max RSS from /usr/bin/time -v over setup + 1 forward.
"""
import argparse
import json
import math
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
PY = sys.executable
THRESHOLDS = [500, 5000, 50000, 500000, 5000000]
SLOTS_PER_CT = 2**15

COMMON_ENV = {"ORION_SLICE_HEIGHT": "1", "ORION_SLICE_MIN_SAVING": "4096",
              "PYTHONPATH": f"{REPO}:{HERE}"}
GPU_ENV = {"ORION_CHEDDAR_WORD32": "1", "ORION_CHEDDAR_MANAGED_MEMORY": "0",
           "ORION_CHEDDAR_BOOT_MIN_KS": "0"}
CPU_ENV = {"GOMAXPROCS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


# ---------------------------------------------------------------- CPU cell
def cpu_cell(threshold, out, trials=0):
    """Lattigo setup plus one forward (plus `trials` timed forwards), in its
    own process. table2.py uses trials=0 (memory only); batch1.py times it."""
    import resource
    import time

    import torch
    import yaml

    import orion
    from bench_sweep import SPARSE_SIZES, _impose_block_diagonal, time_stages
    from dlrm import HELRM

    torch.set_num_threads(1)
    cfg = yaml.safe_load(open(join(HERE, "params.yaml")))
    assert cfg["orion"]["backend"] == "lattigo"
    cfg["orion"]["debug"] = False
    cfg["orion"]["batch_size"] = 1

    torch.manual_seed(0)
    model = HELRM(SPARSE_SIZES, threshold, 4)
    _impose_block_diagonal(model, threshold, 4)
    model.eval()
    dense_in, expanded_sparse = model.fhe_input(
        torch.rand(13),
        torch.tensor([torch.randint(0, s, (1,)).item() for s in SPARSE_SIZES]))
    with torch.no_grad():
        clear = model(dense_in, expanded_sparse)

    orion.init_scheme(cfg)
    orion.fit(model, [dense_in, expanded_sparse])
    input_level = orion.compile(model)
    enc_dense = orion.encrypt(orion.encode(dense_in, input_level))
    enc_sparse = orion.encrypt(orion.encode(expanded_sparse, input_level))
    model.he()
    if trials:
        time_stages(model)

    # One forward: the peak can land inside it, and its output checks the run.
    times, stage_trials = [], []
    for i in range(trials + 1):
        model.stage_seconds = {}
        t0 = time.time()
        out_ct = model(enc_dense, enc_sparse)
        dt = time.time() - t0
        print(f"[timing] {'warmup' if i == 0 else f'trial{i}'}: {dt:.3f}s", flush=True)
        if i:
            times.append(dt)
            stage_trials.append(dict(model.stage_seconds))
            print("[stages] " + " ".join(f"{k}={v:.3f}s" for k, v in model.stage_seconds.items()),
                  flush=True)
    fhe = torch.as_tensor(out_ct.decrypt().decode()).reshape(-1)[:1]
    mae = float((clear.reshape(-1) - fhe).abs().mean())

    json.dump({
        "backend": "lattigo", "compress_threshold": threshold, "batch_size": 1,
        "sparse_width": int(expanded_sparse.shape[1]),
        "input_level": int(input_level), "sparse_level": model.sparse.level,
        "mae": mae, "precision": None if mae == 0 else -math.log2(mae),
        "runtime_seconds": sum(times) / len(times) if times else None,
        "runtime_seconds_trials": times,
        "stage_seconds": ({k: sum(d[k] for d in stage_trials) / len(stage_trials)
                           for k in stage_trials[0]} if stage_trials else None),
        "stage_seconds_trials": stage_trials,
        "peak_rss_kb_ru_maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "GOMAXPROCS": os.environ.get("GOMAXPROCS"),
        "env_knobs": {k: v for k, v in sorted(os.environ.items()) if k.startswith("ORION_")},
        "params": dict(cfg["ckks_params"], boot_LogP=cfg["boot_params"]["LogP"]),
    }, open(out, "w"), indent=1)


# ---------------------------------------------------------------- driver
def run(cmd, env, log):
    with open(log, "w") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        return subprocess.run(cmd, env={**os.environ, **env}, stdout=f,
                              stderr=subprocess.STDOUT).returncode


def time_v_rss_kb(log):
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", open(log).read())
    return int(m.group(1)) if m else None


def sparse_level_from_log(log):
    m = re.search(r"\bsparse @ level=(\d+)", open(log).read())
    return int(m.group(1)) if m else None


def fmt_compression(c):
    mag = int(math.floor(math.log10(c)))
    return f"{round(c, 3 - mag):,.{max(0, 3 - mag)}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=expanduser("~/cheddar-orion-results/table2"))
    ap.add_argument("--cpu-cell", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--trials", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.cpu_cell:
        cpu_cell(args.cpu_cell, args.out, args.trials)
        return

    from bench_sweep import SPARSE_SIZES
    from dlrm import dense_features, expanded_width

    gpu_dir, cpu_dir = join(args.results, "gpu"), join(args.results, "cpu")
    os.makedirs(gpu_dir, exist_ok=True)
    os.makedirs(cpu_dir, exist_ok=True)
    total = sum(SPARSE_SIZES)
    notes = []

    # Level the sparse layer consumes the selector at (min upload level);
    # verified below against what compile() actually assigned.
    gpu_level = {500: 4, 5000: 4, 50000: 4, 500000: 5, 5000000: 5}

    gpu, cpu = {}, {}
    for t in THRESHOLDS:
        js, log = join(gpu_dir, f"threshold_{t}.json"), join(gpu_dir, f"gpu_t{t}.log")
        if not exists(js):
            print(f"[gpu] t={t}", flush=True)
            run([PY, join(HERE, "bench_sweep.py"), "--compress-threshold", str(t),
                 "--base", "4", "--word32", "--seeded-upload", "--batch", "1",
                 "--upload-level", str(gpu_level[t]), "--out-dir", gpu_dir],
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
            run(["/usr/bin/time", "-v", PY, abspath(__file__), "--cpu-cell", str(t),
                 "--out", js], {**COMMON_ENV, **CPU_ENV}, log)
        if exists(js):
            cpu[t] = json.load(open(js))
            cpu[t]["peak_rss_kb"] = time_v_rss_kb(log)
        else:
            notes.append(f"CPU t={t}: run failed, see {log}")

    # ---- checks ----
    for t in THRESHOLDS:
        width = sum(expanded_width(s, t, 4) for s in SPARSE_SIZES)
        if t in cpu and cpu[t]["sparse_width"] != width:
            notes.append(f"CPU t={t}: encrypted width {cpu[t]['sparse_width']} != model width {width}")
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
        if t in cpu and t in gpu and cpu[t]["sparse_level"] != gpu[t]["sparse_level"]:
            notes.append(f"t={t}: sparse layer at level {cpu[t]['sparse_level']} on CPU, "
                         f"{gpu[t]['sparse_level']} on GPU")

    git = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    hdr = "| Threshold | Compression | Slots | Peak RAM (GiB) |\n|---:|---:|---:|---:|"

    def row(t, peak):
        width = sum(expanded_width(s, t, 4) for s in SPARSE_SIZES)
        return f"| {t:,} | {fmt_compression(total / width)}x | {width + dense_features:,} | {peak} |"

    lines = [
        "# Table 2: compression ratio and resource requirements per threshold", "",
        f"Regenerated {date.today().isoformat()} at commit {git} by `table2.py`: current compiler "
        "(sliced diagonal packing), batch 1.",
        f"Compression = {total:,} total categories / selector width; slots = selector width + "
        f"{dense_features} dense inputs. Both are set by the model and identical on the two backends "
        "(the encrypted widths were checked against the model on each run).", "",
        "## CPU: Lattigo, 64-bit chain (params.yaml), single thread", "",
        "Peak RAM is the process max RSS (`/usr/bin/time -v`) over setup and one forward pass.", "",
        hdr,
    ]
    for t in THRESHOLDS:
        c = cpu.get(t)
        lines.append(row(t, f"{c['peak_rss_kb'] / 2**20:.1f}" if c else "—"))
    lines += [
        "", "## GPU: Cheddar, word32 chain, seeded upload at the sparse layer's level", "",
        "Peak RAM is the device allocation peak with managed memory off, over setup and "
        "1 warmup + 3 forwards.", "",
        hdr,
    ]
    for t in THRESHOLDS:
        g = gpu.get(t)
        lines.append(row(t, f"{g['peak_device_mb'] / 1024:.1f}" if g else "—"))
    lines += ["", "Checks: " + ("all passed." if not notes else " ".join(notes)), ""]

    md = join(HERE, "table2.md")
    open(md, "w").write("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote {md}")


if __name__ == "__main__":
    main()
