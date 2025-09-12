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
cd examples/
python3 run_lola.py
```
