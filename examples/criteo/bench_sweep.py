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

import torch

import orion
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
    encrypted_sparse = orion.encrypt(orion.encode(expanded_sparse, input_level))

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
        out = model(encrypted_dense, encrypted_sparse)
        t1 = time.time()
        torch.cuda.nvtx.range_pop()
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
