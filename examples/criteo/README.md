# Criteo HE-LRM

Everything behind Section 5 of the paper: the data, the compressed-embedding
DLRM training, the FHE forward on the CPU (Lattigo) and GPU (Cheddar)
backends, the IVE selector on the same backend, and one script per table and
figure. Run every command below from this directory with the repository's
environment active (`uv run python ...` or an activated venv).

## Files

```text
create_dataset.py   raw Criteo train.txt -> processed/*.bin
dlrm.py             Criteo dataset, DLRM, HELRM (the FHE-shaped model), export
train_criteo.py     train a DLRM with base-p digit-decomposed tables above a threshold
run_orion.py        one end-to-end FHE inference of a trained checkpoint, precision check
bench_sweep.py      HE-LRM forward on the GPU at one threshold and batch (latency, memory, stages)
bench_ive_dlrm.py   the same forward with the IVE selector in place of the one-hot lookup
ive_selector.py     IVE (Cheon et al. 2026, Algorithm 1 VecGen + PC-MM) on the Cheddar bindings
params.yaml         CKKS parameters for the CPU backend
params_cheddar.yaml CKKS parameters for the GPU backend (64-bit word)
figstyle.py         one style for every figure

table2.py    peak memory per threshold, CPU and GPU               (Table: compression and resources)
batch1.py    CPU against GPU at batch 1 with stage times          (Table 1)
tableA.py    selector cost by stage, HE-LRM against IVE           (Table 2, Fig. 9)
ladder.py    latency, throughput and working set against batch   (Fig. 10 and the paged rows)
max_qps.py   largest batch that fits and peak QPS per GPU         (numbers behind Fig. 11)
network.py   upload bytes and live-query latency at 100 Mbps/1 Gbps (Fig. 8)
gpu_ceilings.py  memory-vs-batch and peak-QPS figures              (Fig. 7, Fig. 11)
```

The measurement scripts write their raw JSON and logs under `results/<section>/`,
their tables into `PAPER_NUMBERS.md`, `max_qps.md` and `network.md`, and
their figures under `figures/`. All of these are regenerated outputs and are
not part of the repository. A cell whose log already exists is not re-run, so
every script resumes.

## Data

```bash
python create_dataset.py
```

downloads the public Criteo Kaggle display-advertising archive into `raw/`
and writes `processed/{dense,sparse,label,counts}.bin`. Processing an existing
`raw/train.txt` takes roughly 10 minutes; the download is separate.

## Training

`train_criteo.py` trains the DLRM with every embedding table larger than the
threshold replaced by its base-`p` digit-decomposed tables (the
generalised quotient-remainder compression of Section 4.1) and exports the
HELRM checkpoint plus one sample:

```bash
python train_criteo.py --base 4 --compress-threshold 500 --out-dir model
```

The paper's five models use `--base 4` and thresholds 500, 5,000, 50,000,
500,000 and 5,000,000. Other flags: `--epochs` (1), `--batch-size` (128),
`--eval-batch-size` (16384), `--learning-rate` (0.1), `--seed` (123).
Training uses CUDA when it is visible. Outputs:

```text
<out-dir>/model.pt            HELRM state and configuration
<out-dir>/metrics.json        training loss and test AUC
<out-dir>/sample/{dense,sparse,clear}.pt   one validation sample and its cleartext logit
```

## One FHE inference of a trained model

```bash
python run_orion.py                       # CPU, params.yaml, reads model/
python run_orion.py params_cheddar.yaml   # GPU
```

reads `model/model.pt` and the sample, runs the encrypted forward, and writes
`fhe/result.json` with the cleartext logit, the decrypted FHE logit, their
MAE, the precision in bits and the runtime.

## Environment of every measured cell

The scripts set these themselves; they are listed so a cell can be
reproduced by hand.

| Variable | Value | Meaning |
|---|---|---|
| `ORION_CHEDDAR_WORD32` | `1` | 32-bit word build with the rational-rescaling chain `orion/backend/cheddar/params32/bootparam_40.json` |
| `ORION_CHEDDAR_MANAGED_MEMORY` | `0` | hard device-memory cap; `1` only for the one paged row per ladder |
| `ORION_CHEDDAR_BOOT_MIN_KS` | `0` | full bootstrap key set |
| `ORION_CHEDDAR_BIN_CHUNK` | unset (`0`) | allocator bins off, so the pool's footprint equals the working set |
| `ORION_SLICE_HEIGHT`, `ORION_SLICE_MIN_SAVING` | `1`, `4096` | sliced diagonal packing of the compiler |
| `GOMAXPROCS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS` | `1` | CPU cells run on one thread |

Protocol for every timed cell: one warm-up forward, then three timed forwards;
the reported time is their mean and the three values are kept in the JSON.
Memory is the peak of live device bytes (keys, plaintexts, ciphertexts,
temporaries) over setup and the forwards, read from a statistics adaptor above
Cheddar's pool (the `working-set-counter-no-bins` patch); on the CPU it is the
peak live Go heap sampled every 50 ms. GPU cells run one at a time on one GPU.
HE-LRM cells use the seeded upload at the level the compiler assigns the
sparse layer (`--seeded-upload --upload-level min`, or the level recorded for
batch 1). IVE cells use one-pass PC-MM, Algorithm 1 VecGen
and no bootstrap. The FHE cost does not depend on the weights, so the latency
and memory cells build a randomly initialised HELRM at the exact widths of the
trained model; `run_orion.py` is the end-to-end check on a trained checkpoint.

## One command per result

Run in this order on the main GPU (the paper's RTX PRO 6000):

```bash
python table2.py      # results/table2:  peak memory per threshold, CPU and GPU
python batch1.py      # results/batch1:  Table 1, plus the stage times tableA.py reads
python tableA.py      # results/tableA:  Table 2 and figures/selector_cost_vs_compression
python ladder.py      # results/ladder:  the batch ladder, figures/latency_vs_batch
python network.py     # results/network: upload sizes, figures/network_latency
```

For the per-GPU throughput results, run the ladder on each other card into
its own folder and record the card:

```bash
python ladder.py --results results/ladder_a100    # likewise ladder_h100, ladder_3090
nvidia-smi -L > results/ladder_a100/gpu.txt
```

then, back on any machine with all the folders under `results/`:

```bash
python max_qps.py       # max_qps.md: largest batch that fits and peak QPS per card
python gpu_ceilings.py  # figures/memory_vs_batch, peak_qps_vs_compression, peak_qps_bars
```

The folder names `ladder`, `ladder_a100`, `ladder_3090` and `ladder_h100` and
their capacities are fixed in `max_qps.py`'s `CARDS`; edit that list for other
cards. Every script takes `--results DIR` to read or write elsewhere.

## Single cells by hand

```bash
# HE-LRM forward, threshold 50,000, batch 4, GPU
ORION_CHEDDAR_WORD32=1 ORION_CHEDDAR_MANAGED_MEMORY=0 ORION_CHEDDAR_BOOT_MIN_KS=0 \
ORION_SLICE_HEIGHT=1 ORION_SLICE_MIN_SAVING=4096 PYTHONPATH=../..:. \
python bench_sweep.py --compress-threshold 50000 --base 4 --word32 \
    --seeded-upload --upload-level min --batch 4 --out-dir results/manual

# IVE forward, p = 1024, batch 1, GPU
ORION_CHEDDAR_WORD32=1 ORION_CHEDDAR_MANAGED_MEMORY=0 ORION_CHEDDAR_BOOT_MIN_KS=0 \
ORION_SLICE_HEIGHT=1 ORION_SLICE_MIN_SAVING=4096 PYTHONPATH=../..:. \
python bench_ive_dlrm.py --p 1024 --ell 3 --batch 1 --compress-threshold 500 \
    --trials 3 --no-ref --out results/manual/p1024_b1.json

# CPU cell, threshold 500, three timed forwards
GOMAXPROCS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=../..:. \
python table2.py --cpu-cell 500 --trials 3 --out results/manual/cpu_t500.json
```

`bench_sweep.py --stage-times` adds per-stage times (bottom MLP, lookup,
repack, top MLP) to the JSON; `bench_ive_dlrm.py --check` decrypts and
compares the two PC-MM paths on the first trial. IVE's `--ell` per `p`
follows `ladder.py`'s `IVE_ELL`.
