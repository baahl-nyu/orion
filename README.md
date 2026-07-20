# Orion

## Installation
We tested our implementation on `Ubuntu 22.04.5 LTS`. First, install the required dependencies:

```
sudo apt update && sudo apt install -y \
    build-essential git wget curl ca-certificates \
    python3 python3-pip python3-venv \
    unzip pkg-config libgmp-dev libssl-dev
```

Install Go (for Lattigo backend):

```
cd /tmp
wget https://go.dev/dl/go1.22.3.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.22.3.linux-amd64.tar.gz
echo 'export PATH=/usr/local/go/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
go version # go version go1.22.3 linux/amd64
```

### Install Orion

```
git clone https://github.com/baahl-nyu/orion.git
cd orion/
pip install -e .
```

### Run the examples!

```
uv sync
uv run examples/run_lola.py
uv run examples/run_mlp.py
uv run examples/run_resnet.py
```

#### Running with the DeSiLo backend

To run with the [DeSiLo FHE](https://fhe.desilo.dev/latest/) backend instead of Lattigo:

```
uv pip install desilofhe
```

```
uv run examples/run_lola.py configs/lola_desilo.yml
uv run examples/run_mlp.py configs/mlp_desilo.yml
uv run examples/run_resnet.py configs/resnet_desilo.yml
```

> **Note:** ResNet uses bootstrapping and requires ~32 GB RAM. LoLA and MLP run on a standard laptop.

##### CPU vs GPU

The DeSiLo backend runs on CPU by default. To select the device, add a `device` field under the `orion:` block of the config:

```yaml
orion:
  backend: desilo
  device: cpu   # "cpu" (default) or "gpu"
```

GPU mode requires a CUDA-capable NVIDIA GPU and a CUDA-enabled install of `desilofhe` (see the [DeSiLo docs](https://fhe.desilo.dev/latest/) for installation details). To run any of the examples on GPU, set `device: gpu` in the corresponding `*_desilo.yml` config and re-run the same command — no other changes needed.

The Lattigo backend is CPU-only; the `device` field has no effect when `backend: lattigo`.

#### Running oracle tests

```
uv run pytest tests/oracle/ -v -s
```

To run with DeSiLo:
```
uv run pytest tests/oracle/ -v -s --backend=desilo
```

Bootstrap tests are skipped by default. To include them:
```
uv run pytest tests/oracle/ -v -s --backend=desilo -m ""
```
