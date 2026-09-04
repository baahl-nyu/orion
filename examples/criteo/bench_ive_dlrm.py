"""A DLRM forward with the IVE selector in place of HE-LRM's one-hot.

HE-LRM's sparse path is

    sparse_to_top(sparse(one_hot))      # 2.77M-wide one-hot -> 416 -> 432

where `sparse` is a plaintext linear map the client's one-hot selects a
column block from. IVE replaces that whole path: the client sends one
scalar per (field, sub-table), the server derives the independent
vectors with Algorithm 1, and the PCMM against the pre-transformed table
produces the 416 embedding values directly.

Everything else in the model is untouched and shared: the same bottom
MLP, the same `add`, the same top MLP, the same weights. So this
measures exactly what changes.

Slot layout
-----------
Chosen so the selector's output lands where `sparse_to_top` would have
put it, which is what makes the top MLP consumable without a general
permutation. The query ciphertext holds, at slot `16 + 16f + s`, the
scalar for field `f`'s sub-table `s`. After VecGen and the PCMM,
output ciphertext `r` carries field `f`'s embedding coordinate `r` at
those same slots; a rotate-and-add tree of depth ceil(log2 l) collapses
the sub-tables into slot `16 + 16f`, and one rotation by `r` moves
coordinate `r` to `16 + 16f + r`. Summing the 16 rotated ciphertexts
gives a single ciphertext holding all 416 values in slots 16..431 --
byte-for-byte the layout `sparse_to_top` produces.
"""

import argparse
import math
import time

import numpy as np
import torch

import orion
from orion.backend.python.tensors import CipherTensor
from bench_sweep import build_config, SPARSE_SIZES, _impose_block_diagonal
from dlrm import HELRM, embedding_features, sparse_fields
from ive_selector import (Counter, alpha, folded_table, make_const,
                          power_tree, transformed_table, vecgen)

SEED_BYTES = 16


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=256)
    ap.add_argument("--ell", type=int, default=3)
    ap.add_argument("--boot", action="store_true",
                    help="their Figure 1 protocol: level-0 query, server "
                         "bootstraps before VecGen")
    ap.add_argument("--compress-threshold", type=int, default=500,
                    help="only sizes HE-LRM's own sparse layer, which the "
                         "IVE path does not use; kept small so the compile "
                         "is quick. The IVE forward does not depend on it.")
    # Independent inferences packed into the same query ciphertext. The
    # selector's output layout is 432 slots wide (16 x 27), so block b
    # occupies slots [432b, 432b+432). Every slot-varying plaintext (the
    # PCMM table, fold table, constant column, sub-table mask) is
    # replicated at that stride, and the correctness check covers every
    # block. Nothing in the pipeline is per-slot, so this measures
    # whether the paper's batch amortization holds inside the real model.
    ap.add_argument("--batch", type=int, default=1)
    # Free HE-LRM's one-hot reference path before timing. It exists only
    # so both forwards can be compared in one process; it is not part of
    # IVE's design, and at large p and batch its residents are what push
    # the card over. Verified timing-neutral (163.11 vs 163.20 ms at
    # p=1024, batch 2).
    ap.add_argument("--no-ref", action="store_true")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--pcmm", choices=("matmul", "accum"), default="matmul",
                    help="matmul: the whole d x p table in one pass over the "
                         "coordinate ciphertexts, so each is read once "
                         "(Bae et al. CRYPTO'24). accum: one output row per "
                         "call, the previous behaviour.")
    ap.add_argument("--vecgen", choices=("paper", "fold"), default="paper",
                    help="paper: Algorithm 1 as published. fold: contract "
                         "the table onto the p/2 powers so v is never "
                         "materialised (see folded_table).")
    ap.add_argument("--boot-after", type=int, default=None,
                    help="refresh every live power after this many tree "
                         "stages, so the tree can run deeper than the "
                         "chain. Only 2^k powers exist after k stages, so "
                         "an early k is cheap. Required once "
                         "log2(p/2) + 2 + out_level exceeds the chain top.")
    ap.add_argument("--check", action="store_true",
                    help="decrypt both PCMM paths on the first trial and "
                         "report their difference.")
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    p, ell, d = args.p, args.ell, embedding_features
    nb = args.batch
    BLOCK = embedding_features * (sparse_fields + 1)  # 432
    n = p // 2
    depth = int(math.log2(n)) if n > 1 else 0
    torch.manual_seed(0)

    model = HELRM(SPARSE_SIZES, args.compress_threshold, 4)
    _impose_block_diagonal(model, args.compress_threshold, 4)
    model.eval()
    _rows = [
        model.fhe_input(
            torch.rand(13),
            torch.tensor([torch.randint(0, s, (1,)).item()
                          for s in SPARSE_SIZES]))
        for _ in range(nb)
    ]
    dense_in = torch.cat([r[0] for r in _rows], dim=0)
    exp_sparse = torch.cat([r[1] for r in _rows], dim=0)

    orion.init_scheme(build_config(word32=True, batch=nb))
    orion.fit(model, [dense_in, exp_sparse])
    input_level = orion.compile(model)
    model.he()
    from orion.core.orion import scheme
    B = scheme.backend
    scale = scheme.params.get_default_scale()

    # Reference path, to learn the layout the top MLP expects and to give
    # the IVE path something to be checked against.
    ed = orion.encrypt(orion.encode(dense_in, input_level))
    es = orion.encrypt(orion.encode(exp_sparse, input_level))
    ref_sparse = model.sparse_to_top(model.sparse(es))
    dense_path = model.bottom_to_top(model.bottom(ed))
    width = B.GetCiphertextSlots(ref_sparse.ids[0])
    out_level = B.GetCiphertextLevel(ref_sparse.ids[0])
    ref_logit = model.top(model.add(dense_path, ref_sparse)).decrypt().decode()
    ref_logit = torch.as_tensor(ref_logit).reshape(-1)
    ref_shape, ref_on_shape = ref_sparse.shape, ref_sparse.on_shape
    print(f"[dlrm] top-MLP input: 1 ct, {width} slots, level {out_level}")
    print("[dlrm] one-hot reference logits " +
          " ".join(f"{float(g):.6f}" for g in ref_logit[:nb]), flush=True)
    if args.no_ref:
        for _c in list(ref_sparse.ids) + list(es.ids):
            try:
                B.DeleteCiphertext(_c)
            except Exception:
                pass

    # ---- client: one scalar per (field, sub-table) -----------------
    # The query must carry the whole selector circuit AND land at the
    # level the rest of the model runs at: log2(p/2) for the power tree,
    # one for the PCMM, one for the sub-table mask, and out_level for
    # the top MLP. Integrating IVE is what exposes the last term -- a
    # selector benchmarked on its own never has to pay it.
    need = depth + 2 + out_level
    top = scheme.params.get_max_level()
    if args.boot:
        q_level = 0
    elif args.boot_after is not None:
        # The tree is refreshed partway, so the query only has to carry
        # the stages before that point.
        q_level = min(need, top)
    else:
        q_level = need
    if q_level > top:
        raise SystemExit(
            f"query needs level {need} (log2(p/2)={depth} + 2 + "
            f"out_level={out_level}) but the chain tops out at {top}. "
            f"Use --boot-after K.")
    vals = np.zeros(width, dtype=np.complex128)
    rng = np.random.RandomState(0)
    if nb * BLOCK > width:
        raise SystemExit(
            f"batch {nb} needs {nb*BLOCK} slots but the ciphertext has "
            f"{width}; the layout caps at {width // BLOCK}")
    idx = rng.randint(0, p, size=(nb, sparse_fields, ell))
    for b in range(nb):
        for f in range(sparse_fields):
            for s in range(ell):
                vals[b * BLOCK + embedding_features * (f + 1) + s] = \
                    alpha(int(idx[b, f, s]), p)
    # The backend's complex-lane encode takes [re0, im0, re1, im1, ...].
    interleaved = np.stack([vals.real, vals.imag], axis=1).ravel().tolist()
    pt_q = B.EncodeComplex(interleaved, q_level, scale)
    ct_q = B.EncryptSeeded(pt_q, 0x5EED)
    query_bytes = B.GetCiphertextBxBytes(ct_q) + SEED_BYTES
    print(f"[ive] query: 1 ct at level {q_level}, "
          f"{query_bytes/2**20:.4f} MiB (ring N)",
          flush=True)

    # ---- server: tables, transformed offline -----------------------
    tables = [rng.randn(p, d) / math.sqrt(p) for _ in range(sparse_fields)]
    MLs = [transformed_table(M, p) for M in tables]
    # Slot-varying plaintext: slot 16+16f+s carries field f's row.
    def table_pt(k, r, level):
        v = np.zeros(width)
        for b in range(nb):
            for f in range(sparse_fields):
                for s in range(ell):
                    v[b * BLOCK + embedding_features * (f + 1) + s] = \
                        MLs[f][k, r]
        return B.Encode(v.tolist(), level, scale)

    # The fold's table: 2d rows of weights on the p/2 powers, per field,
    # plus one extra column carrying v's constant last coordinate. That
    # column multiplies Enc(1), so it needs no separate plaintext add and
    # the scale bookkeeping stays inside the PC-MM.
    n_pow = p // 2
    if args.vecgen == "fold":
        folded = [folded_table(Mf, p) for Mf in tables]

        def fold_pt(k, r2, level):
            v = np.zeros(width)
            for b in range(nb):
                for f in range(sparse_fields):
                    for s in range(ell):
                        v[b * BLOCK + embedding_features * (f + 1) + s] = \
                            folded[f][0][r2, k]
            return B.Encode(v.tolist(), level, scale)

        def const_col_pt(r2, level):
            # out_r = 2*Re(S_r), so a constant c_r enters as c_r/2 on the
            # P row and 0 on the Q row.
            v = np.zeros(width)
            if r2 % 2 == 0:
                for b in range(nb):
                    for f in range(sparse_fields):
                        for s in range(ell):
                            v[b * BLOCK + embedding_features * (f + 1) + s] = \
                                folded[f][1][r2 // 2] / 2.0
            return B.Encode(v.tolist(), level, scale)

    boot_slots = B.GetCiphertextSlots(ct_q)
    if args.boot or args.boot_after is not None:
        B.NewBootstrapper(0, boot_slots)
    if args.boot:
        probe_src = B.Bootstrap(ct_q, boot_slots)
    else:
        probe_src = ct_q
    tree_kw = dict(boot_after=args.boot_after, boot_slots=boot_slots)
    if args.vecgen == "fold":
        probe, probe_owned, _ = power_tree(B, probe_src, p, Counter(),
                                           **tree_kw)
    else:
        probe, probe_owned = vecgen(B, probe_src, p, Counter())[0], []
    pcmm_level = min(B.GetCiphertextLevel(c) for c in probe)
    if args.vecgen == "fold":
        # Enc(1) for the constant column, at the level and scale the
        # aligned powers arrive with.
        at_level = [c for c in probe
                    if B.GetCiphertextLevel(c) == pcmm_level][0]
        pt_one = B.Encode([1.0] * width, pcmm_level,
                          B.GetCiphertextScale(at_level))
        const_ct = B.Encrypt(pt_one)
        B.DeletePlaintext(pt_one)
        for c in probe:
            if c is not probe_src:
                B.DeleteCiphertext(c)
        for c in probe_owned:
            B.DeleteCiphertext(c)
    else:
        const_ct = make_const(B, probe[0])
        for c in probe:
            if c is not const_ct:
                B.DeleteCiphertext(c)
    if args.boot:
        B.DeleteCiphertext(probe_src)
    if args.vecgen == "fold":
        rows, cols = 2 * d, n_pow + 1
        pts = [[fold_pt(k, r2, pcmm_level) for k in range(n_pow)]
               + [const_col_pt(r2, pcmm_level)] for r2 in range(rows)]
    else:
        rows, cols = d, p
        pts = [[table_pt(k, r, pcmm_level) for k in range(p)]
               for r in range(d)]
    mask = np.zeros(width)
    for b in range(nb):
        for f in range(sparse_fields):
            mask[b * BLOCK + embedding_features * (f + 1)] = 1.0
    mask_pt = B.Encode(mask.tolist(), pcmm_level - 1, scale)
    zero_pt = B.Encode([0.0] * width, pcmm_level - 2, scale)
    # Bae et al.'s reduction in GPU form: the table is pinned once,
    # offline, and the online PC-MM streams the p coordinate ciphertexts
    # once for all d rows instead of once per row. Here the entries are
    # slot-varying (26 fields share a ciphertext, each with its own
    # table), so they stay full plaintexts -- only the ciphertext
    # re-reads are removed, not the table reads.
    pt_table = None
    if args.pcmm == "matmul":
        pt_table = B.PackPlaintextTableNew(
            [pts[r][k] for r in range(rows) for k in range(cols)], rows)
    print(f"[ive] PCMM at level {pcmm_level}; {d*p} table plaintexts offline"
          f"; one-pass={pt_table is not None}", flush=True)

    times = {"bottom": [], "boot": [], "vecgen": [], "pcmm": [], "sum": [],
             "repack": [], "mlp": [], "total": []}
    # The one-hot forward, timed in this same process and build so the
    # two forwards are comparable without cross-run drift.
    ref_times = []
    for t in range(0 if args.no_ref else args.trials + 1):
        torch.cuda.synchronize(); r0 = time.time()
        _ = model.top(model.add(model.bottom_to_top(model.bottom(ed)),
                                model.sparse_to_top(model.sparse(es))))
        torch.cuda.synchronize()
        if t:
            ref_times.append(time.time() - r0)
    ref_fwd = (sum(ref_times) / len(ref_times)) if ref_times else float("nan")
    print(f"[dlrm] one-hot forward (threshold {args.compress_threshold}): "
          f"{ref_fwd*1e3:.2f} ms", flush=True)

    for t in range(args.trials + 1):
        torch.cuda.synchronize(); tm = time.time()
        dense_path = model.bottom_to_top(model.bottom(ed))
        torch.cuda.synchronize(); t0 = time.time()
        src = B.Bootstrap(ct_q, boot_slots) if args.boot else ct_q
        torch.cuda.synchronize(); t1 = time.time()
        if args.vecgen == "fold":
            cts, tree_owned, _ = power_tree(B, src, p, Counter(), **tree_kw)
        else:
            cts, _ = vecgen(B, src, p, Counter(), const_ct=const_ct)
            tree_owned = []
        torch.cuda.synchronize(); t2 = time.time()
        aligned = [B._align_lt_input(c, pcmm_level)[0] for c in cts]
        if args.vecgen == "fold":
            aligned.append(const_ct)      # the constant column
        if t == 0 and args.check and pt_table is not None \
                and args.vecgen == "paper":
            ref = [B.RescaleNew(B.MulPlaintextAccumNew(aligned, pts[r]))
                   for r in range(d)]
            new = [B.RescaleNew(o)
                   for o in B.MulPlaintextMatMulNew(aligned, pt_table)]
            dr = np.array([B.Decode(B.Decrypt(o)) for o in ref])
            dn = np.array([B.Decode(B.Decrypt(o)) for o in new])
            print(f"[check] accum vs matmul max abs diff "
                  f"{np.abs(dr - dn).max():.3e} (magnitude "
                  f"{np.abs(dr).max():.3e})", flush=True)
            for o in ref + new:
                B.DeleteCiphertext(o)
        if pt_table is not None:
            outs = []
            for o in B.MulPlaintextMatMulNew(aligned, pt_table):
                outs.append(B.RescaleNew(o))
                B.DeleteCiphertext(o)
        else:
            outs = []
            for r in range(d):
                acc = B.MulPlaintextAccumNew(aligned, pts[r])
                outs.append(B.RescaleNew(acc))
                B.DeleteCiphertext(acc)
        if args.vecgen == "fold":
            # 2d accumulators -> d answers: one key-free multiply by i,
            # one add, one conjugation, one add. d conjugations against
            # Algorithm 1's p/2.
            pq, outs = outs, []
            for r in range(d):
                P, Q = pq[2 * r], pq[2 * r + 1]
                iq = B.MulImaginaryUnitNew(Q)
                sr = B.AddCiphertextNew(P, iq)
                cj = B.ConjugateNew(sr)
                outs.append(B.AddCiphertextNew(sr, cj))
                for x in (iq, sr, cj, P, Q):
                    B.DeleteCiphertext(x)
        torch.cuda.synchronize(); t3 = time.time()
        # collapse the ell sub-tables of each field into slot 16+16f
        summed = []
        for o in outs:
            acc, shift = o, 1
            while shift < ell:
                rot = B.RotateNew(acc, shift)
                nxt = B.AddCiphertextNew(acc, rot)
                B.DeleteCiphertext(rot)
                if acc is not o:
                    B.DeleteCiphertext(acc)
                acc, shift = nxt, shift * 2
            m = B.MulPlaintextNew(acc, mask_pt)
            if acc is not o:
                B.DeleteCiphertext(acc)
            summed.append(B.RescaleNew(m))
            B.DeleteCiphertext(m)
        torch.cuda.synchronize(); t4 = time.time()
        # coordinate r to slot 16+16f+r, then fold the 16 into one ct.
        # RotateNew's positive k rotates LEFT and a negative k is not a
        # valid key index -- it decrypts to garbage rather than failing --
        # so moving a value up by r is a rotation by width - r.
        acc = None
        for r, sct in enumerate(summed):
            piece = B.RotateNew(sct, width - r) if r else sct
            if acc is None:
                acc = B.AddPlaintextNew(piece, zero_pt)
            else:
                nxt = B.AddCiphertextNew(acc, piece)
                B.DeleteCiphertext(acc)
                acc = nxt
            if piece is not sct:
                B.DeleteCiphertext(piece)
        sparse_ct = B._align_lt_input(acc, out_level)[0]
        torch.cuda.synchronize(); t5 = time.time()
        sparse_path = CipherTensor(scheme, [sparse_ct], ref_shape,
                                   ref_on_shape)
        logit = model.top(model.add(dense_path, sparse_path))
        torch.cuda.synchronize(); t6 = time.time()

        if t == 0:
            # Check the selector output itself, before the top MLP: slot
            # 16+16f+r must hold sum_s tables[f][idx[f,s]][r].
            dec = np.array(B.Decode(B.Decrypt(sparse_ct)))
            want = np.zeros(width)
            for b in range(nb):
                for f in range(sparse_fields):
                    e = sum(tables[f][idx[b, f, s]] for s in range(ell))
                    for r in range(d):
                        want[b * BLOCK + embedding_features * (f + 1) + r] = \
                            e[r]
            err = np.abs(dec[:width] - want).max()
            print(f"[ive] selector output max abs err {err:.3e}  "
                  f"(|want| max {np.abs(want).max():.3e})")
            print(f"[ive] scales: dense {B.GetCiphertextScale(dense_path.ids[0]):.4e} "
                  f"sparse {B.GetCiphertextScale(sparse_ct):.4e}  "
                  f"levels {B.GetCiphertextLevel(dense_path.ids[0])}/"
                  f"{B.GetCiphertextLevel(sparse_ct)}")
            got = torch.as_tensor(logit.decrypt().decode()).reshape(-1)
            print("[ive] IVE-DLRM logits " +
                  " ".join(f"{float(g):.6f}" for g in got[:nb]), flush=True)
        else:
            times["boot"].append(t1 - t0); times["vecgen"].append(t2 - t1)
            times["pcmm"].append(t3 - t2); times["sum"].append(t4 - t3)
            times["repack"].append(t5 - t4); times["mlp"].append(t6 - t5)
            times["bottom"].append(t0 - tm); times["total"].append(t6 - tm)
        for c in cts:
            if c is not const_ct and c is not src and c is not ct_q:
                B.DeleteCiphertext(c)
        for c in tree_owned:
            B.DeleteCiphertext(c)
        for a, c in zip(aligned, cts):
            if a is not c:
                B.DeleteCiphertext(a)
        for s_ in summed:
            B.DeleteCiphertext(s_)
        if acc != sparse_ct:
            B.DeleteCiphertext(acc)
        B.DeleteCiphertext(sparse_ct)
        if args.boot and src != ct_q:
            B.DeleteCiphertext(src)

    a = lambda k: sum(times[k]) / len(times[k])
    print(f"[ive-dlrm] bottom {a('bottom')*1e3:6.2f}  boot {a('boot')*1e3:6.2f}  "
          f"VecGen {a('vecgen')*1e3:6.2f}  PCMM {a('pcmm')*1e3:6.2f}  "
          f"sum {a('sum')*1e3:5.2f}  repack {a('repack')*1e3:5.2f}  "
          f"add+top {a('mlp')*1e3:6.2f}")
    print(f"[ive-dlrm] IVE forward {a('total')*1e3:7.2f} ms   vs   "
          f"one-hot forward {ref_fwd*1e3:7.2f} ms   "
          f"(batch {nb}, p={p}, ell={ell}, query level {q_level})")
    sel = a("boot") + a("vecgen") + a("pcmm") + a("sum") + a("repack")
    print(f"[ive-dlrm] selector {sel*1e3:7.2f} ms total, "
          f"{sel*1e3/nb:7.3f} ms per inference at batch {nb}")
    if args.out:
        import json
        json.dump({k: a(k) for k in times} | {"p": p, "ell": ell,
                  "batch": nb, "no_ref": args.no_ref,
                  "query_bytes": query_bytes, "boot_protocol": args.boot,
                  "pcmm_kind": args.pcmm,
                  "q_level": q_level, "ref_forward_s": ref_fwd,
                  "threshold": args.compress_threshold},
                  open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
