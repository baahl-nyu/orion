#!/usr/bin/env python
"""Network cost of one live query (batch 1): upload size and end-to-end latency.

    python network.py [--results DIR]

A live query is one client's dense features plus its one-hot selector, both
encrypted and uploaded; the server forward then runs on the GPU. This script
reads the measured batch-1 GPU cells under --results (default
examples/criteo/results): `batch1/gpu/` for HE-LRM (batch1.py) and
`tableA/` for IVE (tableA.py), whose JSONs carry the wire size of the
selector upload measured from the ciphertexts' own buffers (seeded: b plus
the seed). The dense-feature ciphertext is common to both designs; its size
is measured once here (encrypt at the compiled input level, seeded) into
`network/dense_ct.json` under --results, and reused afterwards.

Writes examples/criteo/network.md (whole file) and
figures/network_latency.png (and .pdf): per threshold, a 100 Mbps bar over a
1 Gbps bar of live-query latency, stacked as upload time (dense, selector)
plus the RTX PRO 6000 forward; the x axis is linear to 10 s and compressed
3x from 10 to 20 s. Upload time is bytes * 8 / bandwidth; nothing else is
modelled.
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

from table2 import COMMON_ENV, GPU_ENV, PY, THRESHOLDS, fmt_compression  # noqa: E402
from tableA import IVE_COMPRESSION, IVE_P  # noqa: E402

OUT_MD = join(HERE, "network.md")
OUT_FIG = join(HERE, "figures", "network_latency.png")
BANDWIDTHS = [(100e6, "100 Mbps"), (1e9, "1 Gbps")]

PROBE = r'''
import json, sys, torch
import orion
from bench_sweep import build_config, SPARSE_SIZES, _impose_block_diagonal
from dlrm import HELRM
out = {}
for t in json.loads(sys.argv[2]):
    torch.manual_seed(0)
    model = HELRM(SPARSE_SIZES, t, 4); _impose_block_diagonal(model, t, 4); model.eval()
    d, s = model.fhe_input(torch.rand(13), torch.tensor([torch.randint(0, n, (1,)).item() for n in SPARSE_SIZES]))
    orion.init_scheme(build_config(word32=True, batch=1))
    orion.fit(model, [d, s])
    lvl = orion.compile(model)
    from orion.core.orion import scheme
    B = scheme.backend
    ct = orion.encrypt(orion.encode(d, lvl))
    cid = ct.ids[0]
    out[t] = dict(input_level=lvl, ciphertexts=len(ct.ids), slots=B.GetCiphertextSlots(cid),
                  full_bytes=int(B.GetCiphertextBytes(cid)), seeded_bytes=int(B.GetCiphertextBxBytes(cid)) + 8)
    print(t, out[t], flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
'''


def dense_ct(results):
    """{threshold: {...}} sizes of the dense-feature ciphertext, measured once."""
    js = join(results, "network", "dense_ct.json")
    if not exists(js):
        os.makedirs(dirname(js), exist_ok=True)
        log = js[:-5] + ".log"
        with open(log, "w") as f:
            subprocess.run([PY, "-c", PROBE, js, json.dumps(THRESHOLDS)], cwd=HERE,
                           env={**os.environ, **COMMON_ENV, **GPU_ENV}, stdout=f, stderr=subprocess.STDOUT, check=True)
    return {int(k): v for k, v in json.load(open(js)).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=join(HERE, "results"))
    args = ap.parse_args()
    from bench_sweep import SPARSE_SIZES
    from dlrm import expanded_width
    total = sum(SPARSE_SIZES)

    dense = dense_ct(args.results)
    problems = []
    sizes = {d["seeded_bytes"] for d in dense.values()}
    if len(sizes) != 1 or len({d["input_level"] for d in dense.values()}) != 1:
        problems.append(f"dense ciphertext differs across thresholds: {dense}")
    dense_bytes = dense[THRESHOLDS[0]]["seeded_bytes"]
    dense_level = dense[THRESHOLDS[0]]["input_level"]

    helrm = {}
    for t in THRESHOLDS:
        js = join(args.results, "batch1", "gpu", f"threshold_{t}.json")
        if not exists(js):
            problems.append(f"HE-LRM t={t}: missing {js}")
            continue
        d = json.load(open(js))
        if not d.get("seeded_upload"):
            problems.append(f"HE-LRM t={t}: upload not seeded")
        helrm[t] = dict(comp=total / sum(expanded_width(s, t, 4) for s in SPARSE_SIZES), cts=d["upload_ciphertexts"],
                        level=d["upload_level"], bytes=d["upload_bytes"], fwd=d["runtime_seconds"])
    ive = {}
    for p in IVE_P:
        js = join(args.results, "tableA", f"p{p}_b1.json")
        if not exists(js):
            problems.append(f"IVE p={p}: missing {js}")
            continue
        d = json.load(open(js))
        ive[p] = dict(comp=IVE_COMPRESSION[p], cts=1, level=d["q_level"], bytes=d["query_bytes"], fwd=d["total"])

    def up(b, bw):
        return b * 8 / bw

    # ---- figure: one column-width panel, a 100 Mbps bar over a 1 Gbps bar per threshold, named inside the axes
    # per threshold, each stacked dense upload + selector upload + forward. The x axis is linear to
    # 10 s and 10-20 s is compressed 3x so the 16.8 s bar fits without squeezing the sub-second ones.
    import matplotlib.pyplot as plt
    import numpy as np
    from figstyle import COL_W, FILL, apply, save
    apply()
    rows = [(t, helrm[t]) for t in THRESHOLDS if t in helrm]
    c_fwd, c_dense, c_sel = FILL[2], FILL[4], FILL[4]  # both uploads in teal, the dense one hatched; compute in orange
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    h, gap = 0.38, 0.03
    groups = np.arange(len(rows))[::-1] * 1.0  # smallest threshold at the top
    bar = dict(height=h, edgecolor="black", linewidth=0.5, zorder=2)
    K, KNEE = 1 / 3, 10.0
    minor_y, minor_l = [], []
    for k, (bw, name) in enumerate(BANDWIDTHS):
        ys = groups + ((1 - k) - 0.5) * (h + gap)  # 100 Mbps above 1 Gbps
        dn = [up(dense_bytes, bw) for _ in rows]
        sel = [up(r["bytes"], bw) for _, r in rows]
        fwd = [r["fwd"] for _, r in rows]
        lab = k == 0
        ax.barh(ys, dn, color=c_dense, hatch="///",
                label=f"dense upload ({dense_bytes / 1e6:.1f} MB, fixed)" if lab else None, **bar)
        ax.barh(ys, sel, left=dn, color=c_sel, label="selector upload" if lab else None, **bar)
        ax.barh(ys, fwd, left=[a + b for a, b in zip(dn, sel)], color=c_fwd,
                label="inference compute cost" if lab else None, **bar)
        for y, a, b, c in zip(ys, dn, sel, fwd):
            tot = a + b + c
            if tot > KNEE:  # in the compressed region: label inside the bar's end
                ax.text(tot - 0.5, y, f"{tot:.2f} s", va="center", ha="right", fontsize=5.8)
            else:
                ax.text(tot + 0.12, y, f"{tot:.2f} s", va="center", ha="left", fontsize=5.8)
        minor_y += list(ys)
        minor_l += [name] * len(ys)
    fw = lambda x: np.where(x <= KNEE, x, KNEE + (x - KNEE) * K)  # noqa: E731
    iv = lambda y: np.where(y <= KNEE, y, KNEE + (y - KNEE) / K)  # noqa: E731
    ax.set_xscale("function", functions=(fw, iv))
    ax.set_xlim(0, 20.0)
    ax.set_xticks([0, 2, 4, 6, 8, 10, 20])
    ax.set_yticks(minor_y, minor=True)  # bandwidth on the left of each bar
    ax.set_yticklabels(minor_l, minor=True, fontsize=4.8)
    ax.tick_params(axis="y", which="minor", length=0, pad=1)
    ax.set_yticks(groups)
    ax.tick_params(axis="y", which="major", length=0, labelleft=False)
    for g, (t, r) in zip(groups, rows):  # threshold and compression after the 1 Gbps total of each pair
        y = g - (h + gap) / 2
        tot = up(dense_bytes + r["bytes"], 1e9) + r["fwd"]
        ax.annotate(f"t={t:,} ({fmt_compression(r['comp'])}$\\times$)", (tot + 0.12, y), xytext=(24, 0),
                    textcoords="offset points", fontsize=5.3, color="0.25", ha="left", va="center", zorder=4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Live-query latency (s)")
    ax.set_ylim(-0.55, len(rows) - 0.45)
    ax.legend(loc="upper right", fontsize=5.5, handlelength=1.3, borderpad=0.3, handletextpad=0.4, labelspacing=0.25)
    fig.tight_layout()
    os.makedirs(dirname(OUT_FIG), exist_ok=True)
    save(fig, "network_latency")
    plt.close(fig)

    # ---- markdown
    git = os.popen(f"git -C {REPO} rev-parse --short HEAD").read().strip()
    res = os.path.relpath(args.results, REPO)
    L = ["# Network cost of a live query (batch 1)", "",
         f"Regenerated {date.today().isoformat()} at commit {git} by `network.py` from `{res}/`. A live query is one "
         "client's 13 dense features and its one-hot selector, encrypted and uploaded; upload sizes are measured from the "
         "ciphertexts' own buffers on the word32 chain, seeded (the client sends b and the seed of a, "
         f"{dense_bytes / 1e6:.2f} MB for the dense ciphertext, see below). Forward times are the RTX PRO 6000 batch-1 "
         "cells of `batch1.py` (HE-LRM, seeded upload at the sparse layer's level) and `tableA.py` (IVE, one-pass PC-MM, "
         "Algorithm 1 VecGen). Upload time is bytes x 8 / bandwidth; latency = dense upload + selector upload + forward. "
         "Nothing else (RTT, the 1-ciphertext logit download) is modelled.", "",
         "## Dense-feature ciphertext (common to both designs)", "",
         f"One ciphertext at the compiled input level {dense_level} (the top of the chain) at every threshold: "
         f"{dense[THRESHOLDS[0]]['full_bytes'] / 1e6:.2f} MB unseeded, {dense_bytes / 1e6:.2f} MB seeded. "
         f"Measured by `network/dense_ct.json` (encrypt the dense input at `orion.compile`'s level, "
         "`GetCiphertextBxBytes` + 8-byte seed). HE-LRM and IVE share the dense path, so this term is identical for "
         "both and every design that keeps the bottom MLP.", "",
         "## HE-LRM selector upload and live-query latency", "",
         "| threshold | compression | selector cts | level | selector (MB) | total upload (MB) | fwd (s) | "
         + " | ".join(f"upload @ {n} (s)" for _, n in BANDWIDTHS) + " | "
         + " | ".join(f"latency @ {n} (s)" for _, n in BANDWIDTHS) + " |",
         "|---|---:|---:|---:|---:|---:|---:|" + "---:|" * (2 * len(BANDWIDTHS))]
    for t, r in rows:
        tot = r["bytes"] + dense_bytes
        L.append(f"| t={t:,} | {fmt_compression(r['comp'])}x | {r['cts']} | {r['level']} | {r['bytes'] / 1e6:.2f} | "
                 f"{tot / 1e6:.2f} | {r['fwd']:.3f} | "
                 + " | ".join(f"{up(tot, bw):.3f}" for bw, _ in BANDWIDTHS) + " | "
                 + " | ".join(f"{up(tot, bw) + r['fwd']:.3f}" for bw, _ in BANDWIDTHS) + " |")
    L += ["", f"Figure: `figures/network_latency.png` (the table's latency columns, split into dense upload, selector "
          "upload and forward; one bar per bandwidth and threshold, x axis linear to 10 s and compressed 3x from "
          "10 to 20 s).", "",
          "## IVE selector upload at batch 1, for comparison", "",
          "IVE's query is one ciphertext at every p (432 slots used of 32,768), but it must enter at level "
          "log2(p/2) + 2 + the top-MLP level to carry the power tree, so it is larger than HE-LRM's single "
          "level-4 ciphertext. Same dense ciphertext on top.", "",
          "| p | compression | level | selector (MB) | total upload (MB) | fwd (s) | "
          + " | ".join(f"latency @ {n} (s)" for _, n in BANDWIDTHS) + " |",
          "|---|---:|---:|---:|---:|---:|" + "---:|" * len(BANDWIDTHS)]
    for p in IVE_P:
        if p not in ive:
            continue
        r = ive[p]
        tot = r["bytes"] + dense_bytes
        L.append(f"| p={p} | {fmt_compression(r['comp'])}x | {r['level']} | {r['bytes'] / 1e6:.2f} | {tot / 1e6:.2f} | "
                 f"{r['fwd']:.3f} | " + " | ".join(f"{up(tot, bw) + r['fwd']:.3f}" for bw, _ in BANDWIDTHS) + " |")
    pairs = [(50000, 1024), (500, 32)]
    L += ["", "Matched compression, HE-LRM against IVE (selector bytes, latency at "
          + ", ".join(n for _, n in BANDWIDTHS) + "):", ""]
    for t, p in pairs:
        if t in helrm and p in ive:
            h, i = helrm[t], ive[p]
            L.append(f"* t={t:,} ({fmt_compression(h['comp'])}x) vs p={p} ({fmt_compression(i['comp'])}x): "
                     f"{h['bytes'] / 1e6:.2f} vs {i['bytes'] / 1e6:.2f} MB; "
                     + "; ".join(f"{up(h['bytes'] + dense_bytes, bw) + h['fwd']:.3f} vs "
                                 f"{up(i['bytes'] + dense_bytes, bw) + i['fwd']:.3f} s at {n}" for bw, n in BANDWIDTHS)
                     + ". HE-LRM uploads less and computes faster, so it leads at every bandwidth.")
    L += ["", "Checks: " + ("; ".join(problems) if problems else "all passed") + ".", ""]
    open(OUT_MD, "w").write("\n".join(L))
    print(f"wrote {OUT_MD} and {OUT_FIG}")
    if problems:
        print("Checks:", *problems, sep="\n  ")


if __name__ == "__main__":
    main()
