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
from ive_selector import (Counter, alpha, basis_matrix, folded_table,
                          power_tree, top_powers, transformed_table,
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
    ap.add_argument("--pcmm", choices=("matmul", "accum"), default="matmul",
                    help="matmul: the whole d x p table in one pass over the "
                         "coordinate ciphertexts (Bae et al. CRYPTO'24 -- a "
                         "PC-MM is a plaintext matrix product over the "
                         "ciphertexts' polynomials). accum: one output row "
                         "per call, the previous behaviour.")
    ap.add_argument("--vecgen", choices=("paper", "fold"), default="paper",
                    help="paper: Algorithm 1 as published -- build all p "
                         "coordinates of v, then the PCMM. fold: contract "
                         "the table onto the p/2 powers instead, so v is "
                         "never materialised. Same lookup, same multiply "
                         "count, but p/2 conjugations become d. Not the "
                         "paper's algorithm; see folded_table().")
    ap.add_argument("--stream", type=int, default=0,
                    help="fold only: contract the table into the running "
                         "sums CHUNK columns at a time, producing the "
                         "final tree stage on demand. Those powers are "
                         "never multiplicands, so freeing them as they "
                         "are consumed roughly halves the peak live set. "
                         "0 disables.")
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    p, d = args.p, args.d
    streaming = args.stream > 0 and args.vecgen == "fold" \
        and args.pcmm == "matmul"
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
    if args.vecgen == "fold":
        probe, probe_owned, _ = power_tree(backend, probe_src, p, Counter())
    else:
        probe, probe_owned = vecgen(backend, probe_src, p, Counter())[0], []
    out_level = min(backend.GetCiphertextLevel(c) for c in probe)
    # Capture the coordinates' own scale before the probe is freed. On
    # the word32 rational-rescaling chain a ciphertext's scale drifts
    # from the nominal 2^LogScale, and the table constants must be
    # encoded at the same scale or the product's declared scale is wrong
    # by that ratio and the decode is off.
    out_scale = backend.GetCiphertextScale(probe[0])
    # Algorithm 1's line 16 is a query-independent constant; build it once.
    if args.vecgen == "fold":
        # The constant coordinate is folded into `cst`, so the fold needs
        # no Enc(sqrt(2)) at all -- one fewer live ciphertext.
        const_ct = None
        for c in probe:
            if c is not probe_src:
                backend.DeleteCiphertext(c)
        for c in probe_owned:
            backend.DeleteCiphertext(c)
    else:
        from ive_selector import make_const
        const_ct = make_const(backend, probe[0])
        for c in probe:
            if c is not const_ct:
                backend.DeleteCiphertext(c)
    # The transformed table enters PCMM as d*p broadcast scalars. Held as
    # Cheddar Constants (one word per prime) rather than full plaintexts
    # (N words per prime): same arithmetic, 65536x less memory, and it is
    # what lets p run past 4096 at all -- see MulConstantAccumNew.
    if args.vecgen == "fold":
        W, cst = folded_table(M, p)          # (2d, p/2) and (d,)
        rows, cols = W.shape
        table = [[backend.EncodeConstantNew(float(W[r, k]), out_level,
                                            out_scale)
                  for k in range(cols)] for r in range(rows)]
    else:
        rows, cols = d, p
        table = [[backend.EncodeConstantNew(float(ML[k, r]), out_level,
                                            out_scale)
                  for k in range(p)] for r in range(d)]
    print(f"[ive] {'powers' if args.vecgen == 'fold' else 'VecGen output'} "
          f"at level {out_level}; {rows*cols} table constants encoded "
          f"offline (broadcast scalars, no plaintexts)", flush=True)

    # Bae et al.'s reduction, in the form that matters on a GPU: pack the
    # table prime-major and contiguous so the online PC-MM is a single
    # GEMM over the ciphertexts' RNS limbs -- every coordinate ciphertext
    # read once, all d rows accumulated together -- instead of d separate
    # streaming reductions. Packing is offline; the table is the server's.
    table_id, chunks = None, None
    if streaming:
        # One packed table per column chunk: the first half indexes the
        # multiplicand powers, the rest the ones produced on demand.
        half = (p // 2) // 2
        chunks = []
        for c0 in range(0, half, args.stream):
            chunks.append(list(range(c0, min(c0 + args.stream, half))))
        for c0 in range(half, cols, args.stream):
            chunks.append(list(range(c0, min(c0 + args.stream, cols))))
        chunk_ids = [backend.PackConstantTableNew(
            [table[r][k] for r in range(rows) for k in ch], rows)
            for ch in chunks]
        print(f"[ive] PC-MM: streamed in {len(chunks)} chunks of "
              f"<={args.stream} columns; the top {cols - half} powers are "
              f"built and freed a chunk at a time", flush=True)
    elif args.pcmm == "matmul":
        table_id = backend.PackConstantTableNew(
            [table[r][k] for r in range(rows) for k in range(cols)], rows)
        print(f"[ive] PC-MM: one pass, {rows}x{cols} table packed offline",
              flush=True)
    else:
        print("[ive] PC-MM: one accumulation per output row", flush=True)

    if args.boot:
        backend.DeleteCiphertext(probe_src)

    # One slot per (query, field) group survives the rotate-and-add tree;
    # the mask clears the partial sums the tree leaves in the others.
    mask_pt = backend.Encode(
        [1.0 if i % args.ell == 0 else 0.0 for i in range(slots)],
        out_level - 1, scale)

    times = {"boot": [], "vecgen": [], "pcmm": [], "recomb": [],
             "sum": []}
    for t in range(args.trials + 1):
        torch.cuda.synchronize()
        tb = time.time()
        if args.boot:
            src = backend.Bootstrap(ct_q, boot_slots)
        else:
            src = ct_q
        torch.cuda.synchronize()
        t0 = time.time()
        if args.vecgen == "fold":
            cts, tree_owned, counts = power_tree(
                backend, src, p, Counter(), last_stage=not streaming)
        else:
            cts, counts = vecgen(backend, src, p, Counter(),
                                 const_ct=const_ct)
            tree_owned = []
        torch.cuda.synchronize()
        t1 = time.time()
        aligned = None
        if streaming:
            # Contract chunk by chunk into un-rescaled running sums, and
            # build the final tree stage inside the loop so those powers
            # never all exist at once. Accumulation is linear, so the
            # chunking is exact.
            half = n // 2
            acc = None
            for ch, cid in zip(chunks, chunk_ids):
                temps = []
                if ch[0] < half:
                    srcs = []
                    for k in ch:
                        x, mine = backend._align_lt_input(cts[k], out_level)
                        srcs.append(x)
                        if mine:
                            temps.append(x)
                else:
                    prods, powned = top_powers(
                        backend, cts, n, [k - half + 1 for k in ch], counts)
                    temps.extend(powned)
                    srcs = []
                    for x in prods:
                        y, mine = backend._align_lt_input(x, out_level)
                        srcs.append(y)
                        if mine:
                            temps.append(y)
                    temps.extend(prods)
                part = backend.MulConstantMatMulNew(srcs, cid)
                if acc is None:
                    acc = part
                else:
                    for r in range(rows):
                        backend.AddCiphertext(acc[r], part[r])
                        backend.DeleteCiphertext(part[r])
                for x in temps:
                    backend.DeleteCiphertext(x)
            outs = [backend.RescaleNew(a) for a in acc]
            for a in acc:
                backend.DeleteCiphertext(a)
        elif table_id is not None:
            aligned = [backend._align_lt_input(c, out_level)[0] for c in cts]
            outs = []
            for raw in backend.MulConstantMatMulNew(aligned, table_id):
                outs.append(backend.RescaleNew(raw))
                backend.DeleteCiphertext(raw)
        else:
            aligned = [backend._align_lt_input(c, out_level)[0] for c in cts]
            outs = []
            for r in range(d):
                acc = backend.MulConstantAccumNew(aligned, table[r])
                outs.append(backend.RescaleNew(acc))
                backend.DeleteCiphertext(acc)
        torch.cuda.synchronize()
        t2 = time.time()
        # The fold's PCMM emits P_r and Q_r rather than the answer. One
        # key-free multiply by i, one add, one conjugation and one more
        # add turn each pair into out_r = 2*Re(P_r + i*Q_r) -- d
        # conjugations in total, against Algorithm 1's p/2.
        if args.vecgen == "fold":
            pq, outs = outs, []
            for r in range(d):
                P, Q = pq[2 * r], pq[2 * r + 1]
                iq = backend.MulImaginaryUnitNew(Q)
                sr = backend.AddCiphertextNew(P, iq)
                cj = backend.ConjugateNew(sr)
                counts.conj += 1
                tot = backend.AddCiphertextNew(sr, cj)
                counts.add += 2
                outs.append(backend.AddScalarNew(tot, float(cst[r])))
                counts.add += 1
                for x in (iq, sr, cj, tot, P, Q):
                    backend.DeleteCiphertext(x)
        torch.cuda.synchronize()
        t2b = time.time()
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
            times["recomb"].append(t2b - t2)
            times["sum"].append(t3 - t2b)
        else:
            got = np.array([backend.Decode(backend.Decrypt(o))[:4]
                            for o in outs])
            want = np.array([M[idx[i]] for i in range(4)]).T
            err = np.abs(got - want).max()
            prec = -math.log2(err) if err > 0 else float("inf")
            print(f"[ive] counts   {counts.as_dict()}")
            if args.vecgen == "fold":
                print(f"[ive] Table 2  {counts.expected(p)}  "
                      f"(NOT matched: the fold is not their Algorithm 1 -- "
                      f"it trades p/2 conjugations for {d})")
            else:
                print(f"[ive] Table 2  {counts.expected(p)}")
            print(f"[ive] lookup max abs err {err:.3e}  ({prec:.2f} bits)",
                  flush=True)
        if aligned is not None:
            for a, c in zip(aligned, cts):
                if a != c:
                    backend.DeleteCiphertext(a)
        for o in outs:
            backend.DeleteCiphertext(o)
        for c in cts:
            if c is not None and c is not const_ct and c is not src \
                    and c is not ct_q:
                backend.DeleteCiphertext(c)
        for c in tree_owned:
            backend.DeleteCiphertext(c)

    avg = lambda k: sum(times[k]) / len(times[k]) if times[k] else 0.0
    bt, vg, pc, sm = avg("boot"), avg("vecgen"), avg("pcmm"), avg("sum")
    rc = avg("recomb")
    total = bt + vg + pc + rc + sm
    peak_mb = backend.GetPeakDeviceMemoryMB()
    rss_mb = 0.0
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    rss_mb = int(line.split()[1]) / 1024.0
    except OSError:
        pass
    print(f"[ive] peak pool {peak_mb/1024:.2f} GiB  host VmHWM {rss_mb/1024:.2f} GiB "
          f"(managed_memory={os.environ.get('ORION_CHEDDAR_MANAGED_MEMORY', '0')})")
    stage = "powers " if args.vecgen == "fold" else "VecGen "
    print(f"[ive] boot {bt*1000:7.2f} ms  {stage}{vg*1000:7.2f} ms  "
          f"PCMM {pc*1000:7.2f} ms  recombine {rc*1000:6.2f} ms  "
          f"subtable-sum {sm*1000:7.2f} ms  "
          f"TOTAL {total*1000:7.2f} ms   (one {slots}-slot batch, "
          f"ell={args.ell}, query level {level}, vecgen={args.vecgen})")
    if args.out:
        import json
        json.dump({"p": p, "d": d, "slots": slots, "level": level,
                   "depth": depth, "vecgen_s": vg, "pcmm_s": pc,
                   "recomb_s": rc, "sum_s": sm, "boot_s": bt,
                   "vecgen_kind": args.vecgen, "ell": args.ell,
                   "query_bytes": query_bytes, "counts": counts.as_dict(),
                   "pcmm_kind": args.pcmm,
                   "peak_pool_mb": peak_mb, "host_hwm_mb": rss_mb,
                   "managed_memory": os.environ.get(
                       "ORION_CHEDDAR_MANAGED_MEMORY", "0")},
                  open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
