"""Batch-size=1 HE-LRM latency sweep across compression thresholds.

Reproduces the compression-threshold sweep from Table 2 of the HE-LRM
paper (thresholds 500 / 5,000 / 50,000 / 500,000 / 5,000,000) on the
Cheddar/GPU backend, at batch_size=1. Uses a synthetic (untrained,
randomly-initialized) HELRM instance at the real architecture/widths --
no dataset or training required. Field cardinalities below are the
well-known Criteo Kaggle 26-field vocab sizes (see e.g. the CAFE paper),
so the resulting sparse-table sizes line up with the paper's Table 2
slot counts almost exactly.

Requires the Cheddar backend already built (see
orion/backend/cheddar/README.md) and, for anything beyond the smallest
threshold, the GPU-memory mitigation env vars set below -- without
ORION_CHEDDAR_MANAGED_MEMORY it OOMs on the very first layer's
compile() regardless of the other two flags.

Usage (run one threshold at a time, sequentially -- see BENCH_SWEEP.md):
    python bench_sweep.py --compress-threshold 500
    python bench_sweep.py --compress-threshold 5000000 --out-dir bench_results
"""

import json
import math
import os
import random
import time
from argparse import ArgumentParser
from os import makedirs
from os.path import abspath, dirname, join

import yaml

# Default knob set, re-tuned on the 24GB RTX 3090 AFTER fold-retag
# narrowing (full MIN_KS x SHARED_BS grid, every threshold, both word
# sizes -- see HELRM_ABLATION.md's knob re-sweep): MIN_KS=0 + no
# SHARED_BS wins or ties every cell. The old MIN_KS=1 + SHARED_BS=64
# defaults were tuned pre-fold-retag, when the inflated working set
# (wide top path + 32768-wide boot circuit) made key memory the binding
# constraint; fold-retag removed that, so both memory knobs' compute tax
# (chained boot rotations / ~30-55% extra LT rotations) became pure
# loss. MANAGED_MEMORY=1 remains mandatory on 24GB (thr500 alone peaks
# 23.9GB w32 / 32.5GB w64 at keygen). The memory knobs still exist for
# tighter-memory cards: export ORION_CHEDDAR_BOOT_MIN_KS=1 and/or
# ORION_CHEDDAR_SHARED_BS=64 -- setdefault means any exported env var
# wins over these defaults.
os.environ.setdefault("ORION_CHEDDAR_MANAGED_MEMORY", "1")
os.environ.setdefault("ORION_CHEDDAR_BOOT_MIN_KS", "0")

_WORD32_ACTIVE = os.environ.get("ORION_CHEDDAR_WORD32", "").lower() in (
    "1", "true", "yes", "on")

# Wire cost of one seed. The binding takes a 64-bit seed; a deployment
# wanting 256-bit security margin would send 32 B. Either is negligible
# against a multi-MiB polynomial, so the choice does not move any total.
_SEED_BYTES = 8

import torch

import orion
from orion.backend.python.tensors import CipherTensor
from dlrm import HELRM, embedding_features

# Well-known Criteo Kaggle categorical field cardinalities (26 fields).
# Matches Table 2's slot counts almost exactly at each threshold below
# (e.g. threshold=500 -> sparse_width=1083 vs. the paper's 1096), so
# this is a faithful stand-in for the real dataset's vocab sizes.
SPARSE_SIZES = [
    1461, 583, 10131227, 2202608, 305, 24, 12517, 633, 3, 93145,
    5683, 8351593, 3194, 27, 14992, 5461306, 10, 5652, 2173, 4,
    7046547, 18, 15, 286181, 105, 142572,
]

TABLE2_THRESHOLDS = [500, 5000, 50000, 500000, 5000000]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--compress-threshold", type=int, required=True)
    parser.add_argument("--base", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="bench_results")
    # word32: run on the uint32_t twin backend (ORION_CHEDDAR_WORD32=1 +
    # the pregenerated rational-rescaling chain). The env var must also
    # be set in the launching environment -- bindings.py reads it at
    # import time -- so this flag only swaps in the word32-shaped CKKS
    # params. The LogQ entries are nominal placeholders whose COUNT must
    # match the chain's usable levels (see _word32_num_logq); LogP count
    # is ignored in word32 mode.
    parser.add_argument("--word32", action="store_true")
    # Seeded ("compressed") upload: transmit only the b polynomial plus
    # the 8-byte seed its a was drawn from, and regenerate a server-side.
    # Halves the bytes of every uploaded ciphertext.
    parser.add_argument("--seeded-upload", action="store_true")
    # Upload the selector at a low level and have the server bootstrap
    # it back up before anything else runs. A ciphertext at level L
    # carries L+1 residue limbs, so uploading at level 0 instead of the
    # compiled input level cuts bytes by that ratio -- paid for with one
    # bootstrap per uploaded ciphertext.
    parser.add_argument("--low-level-upload", type=int, default=None,
                        metavar="LEVEL")
    # Upload at an explicit level WITHOUT bootstrapping. Useful because
    # the compiled input level is set by the whole network, while the
    # selector's own consumer (the sparse layer) may run far lower --
    # any limb above that is uploaded only to be dropped.
    parser.add_argument("--upload-level", type=int, default=None,
                        metavar="LEVEL")
    return parser.parse_args()


def _word32_logq_logp_counts():
    """Nominal LogQ/LogP entry counts for the word32 path.

    Neither list's VALUES reach the backend -- _setup_scheme_word32 takes
    every prime from the chain JSON and never reads logp at all. But two
    count constraints bind:

      * len(LogQ) must equal default_encryption_level + 1, or
        _setup_scheme_word32 raises (Orion's level accounting has to agree
        with the native scheme's).
      * len(LogP) <= len(LogQ), enforced by CKKSParameters.__post_init__
        before any backend runs. The historical [31]*12 was fine only
        because the shipped chain declares 17 LogQ entries; against an
        11-entry chain it fails. Deriving it from the chain's auxiliary
        prime count keeps it both legal and meaningful.

    The shipped bootparam_40.json gives (17, 12) -- the previous
    hardcoded pair -- so existing results are unaffected.
    """
    path = os.environ.get("ORION_CHEDDAR_PARAMS_JSON", "").strip()
    if not path:
        return 17, 12
    with open(path) as f:
        chain = json.load(f)
    num_logq = int(chain["default_encryption_level"]) + 1
    num_logp = min(len(chain["auxiliary_primes"]), num_logq)
    return num_logq, num_logp


def _impose_block_diagonal(model, compress_threshold, base):
    """Give the synthetic model the sparsity the real one has.

    HELRM's sparse layer evaluates all 26 embedding lookups as one
    matrix multiply, so its weight is block-diagonal by construction --
    field f's columns feed only field f's 16 output rows, and export()
    builds exactly that with torch.block_diag(*tables).T.

    Constructing HELRM directly, as this script does to avoid needing
    the dataset, instead leaves Linear's kaiming_uniform_ init covering
    every entry. Same shape and therefore the same plaintext count,
    latency and memory -- which is why the ablation numbers are
    unaffected -- but the structure is wrong: a dense matrix has every
    output row live in every column block, and any packing that keys off
    per-block row support sees nothing to exploit.

    Fill the blocks in place (rather than building a second full-size
    tensor) and match FullTable/RadixTable's own init, uniform over
    +-1/sqrt(categories).
    """
    from dlrm import expanded_width

    weight = model.sparse.weight.data          # (sparse_output, sparse_width)
    weight.zero_()
    row = col = 0
    for size in model.sparse_sizes:
        width = expanded_width(size, compress_threshold, base)
        bound = 1.0 / math.sqrt(size)
        weight[row:row + embedding_features,
               col:col + width].uniform_(-bound, bound)
        row += embedding_features
        col += width
    assert row == weight.shape[0] and col == weight.shape[1], (
        f"block-diagonal fill covered {row}x{col}, expected "
        f"{tuple(weight.shape)}")



def _upload_bytes(backend, ct_ids, log_n, seeded=False):
    """Wire size of an uploaded ciphertext list, measured.

    Sizes come from the ciphertext's own polynomial buffers, NOT from
    2*(level+1)*2^LogN*word. That formula holds on the 64-bit chain but
    is wrong on the word32 rational-rescaling chain, where the
    level -> limb map is a lookup table (Parameter::level_config_) and a
    level can carry more than one prime: measured limb counts there are
    2 / 3 / 9 / 11 / 23 at levels 0 / 1 / 5 / 7 / 16, against the
    formula's 1 / 2 / 6 / 8 / 17. A seeded upload transmits only b, so
    it is measured separately rather than assumed to be exactly half.
    """
    total = 0
    for ct_id in ct_ids:
        if seeded:
            total += backend.GetCiphertextBxBytes(ct_id) + _SEED_BYTES
        else:
            total += backend.GetCiphertextBytes(ct_id)
    return int(total)


def build_config(word32=False):
    if word32:
        _nq, _np = _word32_logq_logp_counts()
        ckks = {
            "LogN": 16,
            "LogQ": [40] * _nq,
            "LogP": [31] * _np,
            "LogScale": 40,
            "H": 192,
            "RingType": "standard",
        }
    else:
        ckks = {
            "LogN": 16,
            "LogQ": [55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40],
            "LogP": [61, 61, 61],
            "LogScale": 40,
            "H": 192,
            "RingType": "standard",
        }
    return {
        "ckks_params": ckks,
        "boot_params": {
            "num_cts_levels": 4,
            "num_stc_levels": 3,
        },
        "orion": {
            "batch_size": 1,
            "margin": 2,
            "embedding_method": "hybrid",
            "backend": "cheddar",
            "device": "gpu",
            "fuse_modules": True,
            "debug": False,
            "diags_path": "",
            "keys_path": "",
            "io_mode": "none",
        },
    }


def assert_params_match_yaml():
    """Fail loudly if the benchmark's pinned word64 CKKS/boot params
    drift from the shared example config (params_cheddar.yaml).

    The params are deliberately hard-coded here (batch_size=1, debug
    off, plus the word32 twin set no yaml expresses) so a table row is
    fully determined by this script's git hash -- but that pin means an
    intentional parameter change in the yaml would NOT propagate here.
    This guard turns that silent divergence into an immediate error.
    (The word32 set has no yaml counterpart; it is checked against
    nothing and changes only with the pregenerated rescaling chain.)
    """
    path = join(dirname(abspath(__file__)), "params_cheddar.yaml")
    with open(path, "r", encoding="utf-8") as f:
        ref = yaml.safe_load(f)
    pinned = build_config(word32=False)
    for key in ("ckks_params", "boot_params"):
        if ref.get(key) != pinned[key]:
            raise SystemExit(
                f"bench_sweep.py's pinned {key} no longer match "
                f"params_cheddar.yaml.\n  yaml:   {ref.get(key)}\n"
                f"  pinned: {pinned[key]}\n"
                "If the yaml change is intentional, update build_config "
                "(and re-run the ablation tables); if not, revert the "
                "yaml.")


def main():
    args = parse_args()
    assert_params_match_yaml()
    torch.manual_seed(args.seed)

    model = HELRM(SPARSE_SIZES, args.compress_threshold, args.base)
    _impose_block_diagonal(model, args.compress_threshold, args.base)
    model.eval()

    dense = torch.rand(13)
    sparse_ids = torch.tensor(
        [torch.randint(0, size, (1,)).item() for size in SPARSE_SIZES]
    )
    dense_in, expanded_sparse = model.fhe_input(dense, sparse_ids)

    with torch.no_grad():
        clear_logit = model(dense_in, expanded_sparse)

    if args.word32 and os.environ.get("ORION_CHEDDAR_WORD32", "") not in (
            "1", "true", "yes", "on"):
        raise SystemExit(
            "--word32 requires ORION_CHEDDAR_WORD32=1 in the environment "
            "(bindings.py selects the native module at import time).")
    orion.init_scheme(build_config(word32=args.word32))
    orion.fit(model, [dense_in, expanded_sparse])
    input_level = orion.compile(model)

    encrypted_dense = orion.encrypt(orion.encode(dense_in, input_level))

    # ---- selector upload -------------------------------------------
    # One ciphertext per `slots` selector entries.
    from orion.core.orion import scheme as _scheme
    _slots = _scheme.params.get_slots()
    _log_n = build_config(word32=args.word32)["ckks_params"]["LogN"]

    seeded = args.seeded_upload

    _vec = expanded_sparse
    low_level = args.low_level_upload
    if low_level is not None:
        _upload_level = low_level
    elif args.upload_level is not None:
        _upload_level = args.upload_level
    else:
        _upload_level = input_level
    _plain = orion.encode(_vec, _upload_level)

    if seeded:
        # Fixed RNG so a benchmark cell is reproducible. A real client
        # MUST use a CSPRNG and a fresh seed per ciphertext: reusing an
        # `a` across two messages under one key leaks their difference.
        _seed_rng = random.Random(args.seed)
        upload_seeds = [_seed_rng.getrandbits(63) for _ in _plain.ids]
        _ct_ids = [_scheme.backend.EncryptSeeded(_pid, _sd)
                   for _pid, _sd in zip(_plain.ids, upload_seeds)]
        uploaded_sparse = CipherTensor(_scheme, _ct_ids, _vec.shape)
        # Only b crossed the wire: drop a so the timed region has to
        # rebuild it, and so a bug there cannot silently pass.
        for _cid in _ct_ids:
            _scheme.backend.DropAx(_cid)
    else:
        upload_seeds = None
        uploaded_sparse = orion.encrypt(_plain)

    for _pid in _plain.ids:
        _scheme.backend.DeletePlaintext(_pid)

    encrypted_sparse = None if seeded else uploaded_sparse

    # ORION_LANE_DEBUG=1: decrypt the received selector and compare it to
    # the plaintext one, isolating upload-path error from model error.
    if os.environ.get("ORION_LANE_DEBUG"):
        _flat = expanded_sparse.flatten().to(torch.float64)
        if seeded:
            for _cid, _sd in zip(uploaded_sparse.ids, upload_seeds):
                _scheme.backend.ExpandSeededAx(_cid, _sd)
        _probe = uploaded_sparse
        _err = 0.0
        for _i, _cid in enumerate(_probe.ids):
            _got = torch.tensor(_scheme.backend.Decode(
                _scheme.backend.Decrypt(_cid)), dtype=torch.float64)
            _want = _flat[_i*_slots:(_i+1)*_slots]
            _got = _got[:_want.numel()]
            _err = max(_err, (_got - _want).abs().max().item())
        print(f"[lane-debug] input_level={input_level} "
              f"selector max abs err={_err:.3e}", flush=True)
        if seeded:  # re-drop: the probe expanded them
            for _cid in uploaded_sparse.ids:
                _scheme.backend.DropAx(_cid)

    if low_level is not None:
        # One-time, outside the timed region: the network's own
        # bootstrappers are prepared for its internal widths only, and
        # the selector is uploaded at the full slot width.
        _boot_slots = _scheme.backend.GetCiphertextSlots(uploaded_sparse.ids[0])
        _scheme.backend.NewBootstrapper(0, _boot_slots)

    upload_cts = len(uploaded_sparse.ids)
    upload_bytes = _upload_bytes(_scheme.backend, uploaded_sparse.ids,
                                 _log_n, seeded=seeded)
    _tags = (([" seeded"] if seeded else [])
             + ([f" level {_upload_level}"]
                if _upload_level != input_level else []))
    print(f"[upload] selector: {upload_cts} ciphertext(s), "
          f"{upload_bytes/2**20:.1f} MiB"
          f"{(' (' + ','.join(t.strip() for t in _tags) + ')') if _tags else ''}",
          flush=True)

    model.he()
    # 1 warmup + 3 timed trials in the SAME process. The warmup forward
    # absorbs first-touch costs -- UVM paging the key/plaintext working
    # set onto the card, lazy per-width bootstrap-circuit prep -- which
    # otherwise dominate a fresh process's first forward (measured at
    # threshold 500: cold 24s vs warm 3-4s on the same code). Trials
    # report the warm steady state, matching the protocol used for every
    # other benchmark in this repo.
    # Profiling only: ORION_BENCH_FORWARDS=2 runs warmup + one timed
    # forward so nsys can capture a single steady-state pass instead of
    # four (at threshold 5,000,000 a full four-pass UVM trace is
    # unusably large). Unset means 4 -- the protocol every recorded
    # result in HELRM_ABLATION.md was measured under. Anything below 2
    # would leave no warm trial at all.
    n_forwards = max(2, int(os.environ.get("ORION_BENCH_FORWARDS", "4")))

    trial_times = []
    for i in range(n_forwards):
        label = "warmup" if i == 0 else f"trial{i}"
        # NVTX ranges let nsys --capture-range=nvtx profile exactly one
        # forward, excluding compile/keygen/encrypt and the warmup pass
        # that absorbs first-touch paging. No colon in the name: nsys
        # parses --nvtx-capture as range@domain, so a colon can be
        # misread and the range then never matches.
        torch.cuda.nvtx.range_push(f"forward_{label}")
        t0 = time.time()
        # Seed expansion is server-side work on the received upload, so
        # it is timed with the forward.
        if seeded:
            # Server rebuilds each a from its seed. Overwrites ax in
            # place, so it is idempotent across trials.
            torch.cuda.nvtx.range_push("upload_seed_expand")
            for _cid, _sd in zip(uploaded_sparse.ids, upload_seeds):
                _scheme.backend.ExpandSeededAx(_cid, _sd)
            torch.cuda.nvtx.range_pop()
        if low_level is not None:
            torch.cuda.nvtx.range_push("upload_bootstrap")
            # Bootstrap back up to a usable level. Returns fresh
            # handles, so keep them separate from the uploaded ones
            # (which every trial re-reads).
            _booted = [_scheme.backend.Bootstrap(_cid, _boot_slots)
                       for _cid in uploaded_sparse.ids]
            _received = CipherTensor(_scheme, _booted, uploaded_sparse.shape)
            torch.cuda.nvtx.range_pop()
        else:
            _received = uploaded_sparse
        sparse_in = _received
        torch.cuda.nvtx.range_push("model_forward")
        out = model(encrypted_dense, sparse_in)
        torch.cuda.nvtx.range_pop()
        t1 = time.time()
        torch.cuda.nvtx.range_pop()
        if low_level is not None:
            for _cid in _received.ids:
                _scheme.backend.DeleteCiphertext(_cid)
        print(f"[timing] {label}: {t1 - t0:.3f}s", flush=True)
        if i > 0:
            trial_times.append(t1 - t0)
    runtime = sum(trial_times) / len(trial_times)
    fhe_logit = out.decrypt().decode()
    fhe_logit = torch.as_tensor(fhe_logit).detach().cpu()

    mae = float((clear_logit - fhe_logit).abs().mean())
    precision = None if mae == 0 else -math.log2(mae)

    # Memory counters. peak_device_mb is the true *logical* peak from the
    # managed-memory patch's statistics adaptor -- it counts bytes allocated
    # regardless of whether they were physically resident or paged to host
    # RAM by UVM, so it stays meaningful under MANAGED_MEMORY=1. Without it
    # a memory ablation can't distinguish "fits in VRAM" from "fits only
    # because UVM paged it out". Cheddar-only; None on other backends.
    peak_device_mb = key_memory_mb = None
    try:
        from orion.core.orion import scheme as _scheme
        _backend = _scheme.backend
        if hasattr(_backend, "GetPeakDeviceMemoryMB"):
            peak_device_mb = _backend.GetPeakDeviceMemoryMB()
            key_memory_mb = _backend.GetKeyMemoryMB()
    except Exception as exc:  # never fail a timed run over instrumentation
        print(f"[mem] counters unavailable: {exc}", flush=True)

    # Record the knobs actually in effect so each JSON is self-describing
    # and a grid cell can be identified from its output alone.
    env_knobs = {
        k: v for k, v in sorted(os.environ.items())
        if k.startswith("ORION_CHEDDAR_")
    }

    result = {
        "compress_threshold": args.compress_threshold,
        "base": args.base,
        "batch_size": 1,
        "clear": clear_logit.reshape(-1).tolist(),
        "fhe": fhe_logit.reshape(-1).tolist(),
        "mae": mae,
        "precision": precision,
        "runtime_seconds": runtime,  # mean of warm trials
        "runtime_seconds_trials": trial_times,
        "seeded_upload": seeded,
        "low_level_upload": low_level,
        "upload_level": _upload_level,
        "upload_ciphertexts": upload_cts,
        "upload_bytes": upload_bytes,
        "peak_device_mb": peak_device_mb,
        "key_memory_mb": key_memory_mb,
        "env_knobs": env_knobs,
    }

    makedirs(args.out_dir, exist_ok=True)
    out_path = join(args.out_dir, f"threshold_{args.compress_threshold}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, indent=2) + "\n")

    print(json.dumps(result, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
