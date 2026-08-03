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
from os.path import join

os.environ.setdefault("ORION_CHEDDAR_MANAGED_MEMORY", "1")
os.environ.setdefault("ORION_CHEDDAR_BOOT_MIN_KS", "1")

import torch

import orion
from dlrm import HELRM

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
    # params (17 nominal LogQ entries to match the chain's 16 usable
    # levels; LogP count is ignored in word32 mode).
    parser.add_argument("--word32", action="store_true")
    return parser.parse_args()


def build_config(word32=False):
    if word32:
        ckks = {
            "LogN": 16,
            "LogQ": [40] * 17,
            "LogP": [31] * 12,
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


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    model = HELRM(SPARSE_SIZES, args.compress_threshold, args.base)
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
    start = time.time()
    fhe_logit = model(encrypted_dense, encrypted_sparse).decrypt().decode()
    runtime = time.time() - start
    fhe_logit = torch.as_tensor(fhe_logit).detach().cpu()

    mae = float((clear_logit - fhe_logit).abs().mean())
    precision = None if mae == 0 else -math.log2(mae)

    result = {
        "compress_threshold": args.compress_threshold,
        "base": args.base,
        "batch_size": 1,
        "clear": clear_logit.reshape(-1).tolist(),
        "fhe": fhe_logit.reshape(-1).tolist(),
        "mae": mae,
        "precision": precision,
        "runtime_seconds": runtime,
    }

    makedirs(args.out_dir, exist_ok=True)
    out_path = join(args.out_dir, f"threshold_{args.compress_threshold}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, indent=2) + "\n")

    print(json.dumps(result, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
