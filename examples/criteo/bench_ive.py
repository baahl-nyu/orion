"""Measure the IVE selector's server cost and query size on this GPU.

Runs the paper's Algorithm 1 plus its PCMM lookup on Cheddar, at the
same ring degree, chain and clock as every HE-LRM number in
PAPER_NUMBERS.md, so the two selectors can be compared without scaling
anyone's CPU results.

Reports, per (p, upload level):
  - operation counts, checked against the paper's Table 2
  - VecGen and PCMM wall time for one full N/2-slot batch
  - query bytes, measured from the ciphertext's own buffers
  - correctness: the recovered embedding against the plaintext lookup
"""

import argparse
import math
import os
import time

import numpy as np
import torch

import orion
from bench_sweep import build_config
from dlrm import embedding_features
from ive_selector import (Counter, alpha, basis_matrix, transformed_table,
                          v_of_alpha, vecgen)

SEED_BYTES = 16  # the paper's 128-bit seed


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=256,
                    help="sub-table size; the paper evaluates log p = 6..10")
    ap.add_argument("--d", type=int, default=embedding_features)
    ap.add_argument("--upload-level", type=int, default=None,
                    help="level the query arrives at; default = just enough "
                         "for VecGen's log2(p/2) tree plus the PCMM")
    ap.add_argument("--word32", action="store_true")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--ell", type=int, default=3,
                    help="sub-tables per field whose PCMM outputs must be "
                         "summed. ceil(log_p P) over Criteo's 26 fields "
                         "maxes at 3 for p=256.")
    ap.add_argument("--boot", action="store_true",
                    help="their Figure 1 protocol: the query arrives at "
                         "level 0, so the server bootstraps it before "
                         "VecGen can run. Timed with everything else.")
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    p, d = args.p, args.d
    n = p // 2
    depth = int(math.log2(n)) if n > 1 else 0

    orion.init_scheme(build_config(word32=args.word32))
    from orion.core.orion import scheme
    backend = scheme.backend
    params = scheme.params
    slots = params.get_slots()
    max_level = params.get_max_level()

    # The query must arrive with room for the multiplication tree and the
    # PCMM that follows. Their protocol sends level 0 and ring-packs,
    # which is noise-free and modulus-preserving (their App. A.6), so a
    # level-0 query would have to be bootstrapped before VecGen could
    # run at all -- measured separately by --upload-level 0.
    # depth for the power tree, +1 for the PCMM, +1 for the mask that
    # ends the sub-table summation. The last one is easy to miss: at
    # p=256 a depth+1 query leaves the PCMM output at level 0 and the
    # summation cannot run at all.
    need = depth + 2
    if args.boot:
        level = 0
    else:
        level = args.upload_level if args.upload_level is not None else need
    print(f"[ive] p={p} d={d} slots={slots} tree depth={depth} "
          f"level needed={need} upload level={level} (chain max {max_level})",
          flush=True)

    # ---- client: one scalar per query slot -------------------------
    rng = np.random.RandomState(0)
    idx = rng.randint(0, p, size=slots)
    alphas = np.array([alpha(int(j), p) for j in idx])
    inter = np.empty(2 * slots)
    inter[0::2] = alphas.real
    inter[1::2] = alphas.imag
    scale = params.get_default_scale()
    pt_q = backend.EncodeComplex(inter.tolist(), level, scale)
    seed = 0x5EED
    ct_q = backend.EncryptSeeded(pt_q, seed)
    query_bytes = backend.GetCiphertextBxBytes(ct_q) + SEED_BYTES
    backend.DeletePlaintext(pt_q)
    print(f"[ive] query: 1 ciphertext, {query_bytes/2**20:.4f} MiB "
          f"(b + 128-bit seed, ring N; halve for their ring-N/2 transmission)",
          flush=True)

    # ---- server: the plaintext table, transformed offline ----------
    M = rng.randn(p, d) / math.sqrt(p)
    ML = transformed_table(M, p)          # (p, d)

    # The transformed table M@L is preprocessed offline (their Figure 1),
    # so encoding it is not server inference work. Encode once, at the
    # level VecGen's output lands on, and keep it out of the timed loop.
    if args.boot:
        boot_slots = backend.GetCiphertextSlots(ct_q)
        backend.NewBootstrapper(0, boot_slots)
        probe_src = backend.Bootstrap(ct_q, boot_slots)
    else:
        probe_src = ct_q
    probe, _ = vecgen(backend, probe_src, p, Counter())
    out_level = min(backend.GetCiphertextLevel(c) for c in probe)
    # Algorithm 1's line 16 is a query-independent constant; build it once.
    from ive_selector import make_const
    const_ct = make_const(backend, probe[0])
    for c in probe:
        if c is not const_ct:
            backend.DeleteCiphertext(c)
    pts = [[backend.Encode([float(ML[k, r])] * slots, out_level, scale)
            for k in range(p)] for r in range(d)]
    print(f"[ive] VecGen output at level {out_level}; "
          f"{d*p} table plaintexts encoded offline", flush=True)

    if args.boot:
        backend.DeleteCiphertext(probe_src)

    # One slot per (query, field) group survives the rotate-and-add tree;
    # the mask clears the partial sums the tree leaves in the others.
    mask_pt = backend.Encode(
        [1.0 if i % args.ell == 0 else 0.0 for i in range(slots)],
        out_level - 1, scale)

    times = {"boot": [], "vecgen": [], "pcmm": [], "sum": []}
    for t in range(args.trials + 1):
        torch.cuda.synchronize()
        tb = time.time()
        if args.boot:
            src = backend.Bootstrap(ct_q, boot_slots)
        else:
            src = ct_q
        torch.cuda.synchronize()
        t0 = time.time()
        cts, counts = vecgen(backend, src, p, Counter(), const_ct=const_ct)
        torch.cuda.synchronize()
        t1 = time.time()
        aligned = [backend._align_lt_input(c, out_level)[0] for c in cts]
        outs = []
        for r in range(d):
            acc = backend.MulPlaintextAccumNew(aligned, pts[r])
            outs.append(backend.RescaleNew(acc))
            backend.DeleteCiphertext(acc)
        torch.cuda.synchronize()
        t2 = time.time()
        # IVE's PCMM output is coordinate-sharded across slots, so the
        # ell pieces of one field sit in different slots and the field's
        # embedding does not exist until they are summed. With the group
        # laid out contiguously that is a rotate-and-add tree of depth
        # ceil(log2 ell) on each of the d outputs, plus one mask to keep
        # the slot the tree accumulated into.
        summed = []
        for o in outs:
            acc, shift = o, 1
            while shift < args.ell:
                rot = backend.RotateNew(acc, shift)
                nxt = backend.AddCiphertextNew(acc, rot)
                backend.DeleteCiphertext(rot)
                if acc is not o:
                    backend.DeleteCiphertext(acc)
                acc, shift = nxt, shift * 2
            masked = backend.MulPlaintextNew(acc, mask_pt)
            if acc is not o:
                backend.DeleteCiphertext(acc)
            summed.append(backend.RescaleNew(masked))
            backend.DeleteCiphertext(masked)
        torch.cuda.synchronize()
        t3 = time.time()
        for m in summed:
            backend.DeleteCiphertext(m)
        if args.boot and src != ct_q:
            backend.DeleteCiphertext(src)
        counts.pcmm = 1
        if t > 0:
            times["vecgen"].append(t1 - t0)
            times["boot"].append(t0 - tb)
            times["pcmm"].append(t2 - t1)
            times["sum"].append(t3 - t2)
        else:
            got = np.array([backend.Decode(backend.Decrypt(o))[:4]
                            for o in outs])
            want = np.array([M[idx[i]] for i in range(4)]).T
            err = np.abs(got - want).max()
            prec = -math.log2(err) if err > 0 else float("inf")
            print(f"[ive] counts   {counts.as_dict()}")
            print(f"[ive] Table 2  {counts.expected(p)}")
            print(f"[ive] lookup max abs err {err:.3e}  ({prec:.2f} bits)",
                  flush=True)
        for a, c in zip(aligned, cts):
            if a != c:
                backend.DeleteCiphertext(a)
        for o in outs:
            backend.DeleteCiphertext(o)
        for c in cts:
            if c is not const_ct:
                backend.DeleteCiphertext(c)

    avg = lambda k: sum(times[k]) / len(times[k])
    bt, vg, pc, sm = avg("boot"), avg("vecgen"), avg("pcmm"), avg("sum")
    total = bt + vg + pc + sm
    print(f"[ive] boot {bt*1000:7.2f} ms  VecGen {vg*1000:7.2f} ms  "
          f"PCMM {pc*1000:7.2f} ms  subtable-sum {sm*1000:7.2f} ms  "
          f"TOTAL {total*1000:7.2f} ms   (one {slots}-slot batch, "
          f"ell={args.ell}, query level {level})")
    if args.out:
        import json
        json.dump({"p": p, "d": d, "slots": slots, "level": level,
                   "depth": depth, "vecgen_s": vg, "pcmm_s": pc,
                   "sum_s": sm, "boot_s": bt, "ell": args.ell,
                   "query_bytes": query_bytes, "counts": counts.as_dict()},
                  open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
