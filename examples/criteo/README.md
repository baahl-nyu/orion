# Criteo HELRM

This example trains the Criteo DLRM, exports the trained weights into the
example-local HELRM model, and runs one Lattigo FHE inference. `run_orion.py`
compares the cleartext HELRM logit with the decrypted FHE logit.

## Data

`create_dataset.py` writes processed Criteo files into `processed/`:

```text
dense.bin
sparse.bin
label.bin
counts.bin
```

Fetch the public Criteo archive if `processed/` is missing:

```bash
uv run python create_dataset.py
```

On this machine, creating `processed/` from an existing raw `train.txt` should
take roughly 10-12 minutes. Download time for the public archive is separate
and depends on the network.

## Run

From this directory:

```bash
uv run python create_dataset.py
uv run python train_criteo.py --base 4 --compress-threshold 20000 --out-dir model
uv run python run_orion.py
```

With an activated environment that already has Orion installed:

```bash
python train_criteo.py
python run_orion.py
```

Training uses CUDA when CUDA is visible. FHE inference runs on CPU.

## Files

```text
create_dataset.py  raw Criteo train.txt -> processed/*.bin
dlrm.py            Criteo dataset, DLRM, HELRM, and export
train_criteo.py    train DLRM and save exported HELRM artifacts
run_orion.py       run one Orion/Lattigo FHE inference
params.yaml        Orion scheme parameters
```

## Outputs

```text
model/model.pt
model/metrics.json
model/sample/dense.pt
model/sample/sparse.pt
model/sample/clear.pt
fhe/result.json
```

`fhe/result.json` contains `clear`, `fhe`, `mae`, `precision`, and
`runtime_seconds`.

## Example Result

| Clear | FHE | Precision | Runtime |
| ---: | ---: | ---: | ---: |
| -1.0446 | -1.0448 | 12.49 bits | 204.50 s |
