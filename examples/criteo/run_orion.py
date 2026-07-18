import json
import math
import time
from os import makedirs
from os.path import join

import torch

import orion
from dlrm import HELRM


def compare(clear, fhe):
    mae = float((clear - fhe).abs().mean())
    precision = None if mae == 0 else -math.log2(mae)
    return {
        "clear": clear.reshape(-1).tolist(),
        "fhe": fhe.reshape(-1).tolist(),
        "mae": mae,
        "precision": precision,
    }


def run_fhe(model, dense, expanded_sparse):
    orion.init_scheme("params.yaml")
    orion.fit(model, [dense, expanded_sparse])
    input_level = orion.compile(model)

    encrypted_dense = orion.encrypt(orion.encode(dense, input_level))
    encrypted_sparse = orion.encrypt(orion.encode(expanded_sparse, input_level))

    model.he()
    start = time.time()
    fhe_logit = model(encrypted_dense, encrypted_sparse).decrypt().decode()
    runtime = time.time() - start

    return torch.as_tensor(fhe_logit).detach().cpu(), runtime


def main():
    model_dir = "model"
    out_dir = "fhe"
    makedirs(out_dir, exist_ok=True)

    checkpoint = torch.load(
        join(model_dir, "model.pt"),
        map_location="cpu",
        weights_only=False,
    )
    sample_dir = join(model_dir, "sample")
    dense = torch.load(
        join(sample_dir, "dense.pt"),
        map_location="cpu",
        weights_only=False,
    )
    sparse_ids = torch.load(
        join(sample_dir, "sparse.pt"),
        map_location="cpu",
        weights_only=False,
    )
    clear_logit = torch.load(
        join(sample_dir, "clear.pt"),
        map_location="cpu",
        weights_only=False,
    )

    model = HELRM(
        checkpoint["sparse_sizes"],
        checkpoint["compress_threshold"],
        checkpoint["base"],
    )
    model.load_state_dict(checkpoint["weights"])
    model.eval()
    dense, expanded_sparse = model.fhe_input(dense, sparse_ids)

    fhe_logit, runtime = run_fhe(model, dense, expanded_sparse)
    result = compare(clear_logit, fhe_logit)
    result["runtime_seconds"] = runtime

    with open(join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(result, indent=2) + "\n")

    print()
    print(clear_logit)
    print(fhe_logit)
    print(f"\nMAE: {result['mae']:.4f}")
    if result["precision"] is None:
        print("Precision: inf")
    else:
        print(f"Precision: {result['precision']:.4f}")
    print(f"Runtime: {runtime:.4f} secs.\n")


if __name__ == "__main__":
    main()
