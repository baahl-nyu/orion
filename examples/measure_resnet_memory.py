"""Measure where ResNet20's GPU memory actually goes, phase by phase and
layer by layer, on whichever backend the given config selects.

Replicates Scheme.compile() here (rather than modifying orion/core/orion.py)
so checkpoints can be inserted at each phase boundary:
  1. baseline (after fit, before compile)
  2. after diagonal generation for all LinearTransform layers (cleartext
     packing only -- expected near-zero GPU delta on every backend)
  3. after generating all bootstrappers (boot circuits: FFT tables + their
     required rotation keys, where applicable)
  4. after each layer's compile() in the final loop (per-layer diagonal
     encode + rotation-key generation) -- printed incrementally so an OOM
     (if any) shows exactly where it happened

Reports GetKeyMemoryMB()/GetPeakDeviceMemoryMB() when the backend exposes
them (cheddar only -- see orion/backend/cheddar/ext/cheddar_pybind.cu and
.../MemoryPool.h in the shared cheddar clone), plus nvidia-smi memory.used
for every physical GPU (works on any backend, but under cheddar's
ORION_CHEDDAR_MANAGED_MEMORY it only reflects the VRAM-resident portion --
use GetPeakDeviceMemoryMB for the true logical peak in that case).

Usage:
    python examples/measure_resnet_memory.py configs/resnet_cheddar.yml
    python examples/measure_resnet_memory.py configs/resnet_desilo.yml

    # cheddar-specific knobs (see orion/backend/cheddar/README.md):
    ORION_CHEDDAR_MANAGED_MEMORY=1 ORION_CHEDDAR_BOOT_MIN_KS=1 \\
        python examples/measure_resnet_memory.py configs/resnet_cheddar.yml
"""
import math
import subprocess
import sys

import orion
import orion.models as models
from orion.core.utils import get_cifar_datasets
from orion.core.fuser import Fuser
from orion.core.network_dag import NetworkDAG
from orion.core.auto_bootstrap import BootstrapSolver, BootstrapPlacer
from orion.nn.module import Module
from orion.nn.linear import LinearTransform


def gpu_used_mb_all():
    # nvidia-smi is unaffected by CUDA_VISIBLE_DEVICES (it queries the driver
    # directly), so report every physical GPU's usage and let the caller
    # pick the right one rather than guessing an index.
    out = subprocess.check_output([
        "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"
    ]).decode()
    return [int(v) for v in out.strip().split("\n")]


def checkpoint(label, scheme):
    # GetKeyMemoryMB/GetPeakDeviceMemoryMB are cheddar-only native probes;
    # desilo/lattigo have no equivalent, so fall back to nvidia-smi alone.
    key_mb = scheme.backend.GetKeyMemoryMB() if hasattr(scheme.backend, "GetKeyMemoryMB") else None
    peak_mb = (scheme.backend.GetPeakDeviceMemoryMB()
               if hasattr(scheme.backend, "GetPeakDeviceMemoryMB") else None)
    all_gpus = gpu_used_mb_all()
    gpu_str = ", ".join(f"gpu{i}={v}MB" for i, v in enumerate(all_gpus))
    key_str = f"key_mem={key_mb:8.1f} MB | " if key_mb is not None else ""
    peak_str = f"peak_pool={peak_mb:9.1f} MB | " if peak_mb is not None else ""
    print(f"[CKPT] {label}: {key_str}{peak_str}{gpu_str}", flush=True)
    return key_mb, peak_mb, all_gpus


def instrumented_compile(scheme, net):
    net.set_scheme(scheme)
    net.set_margin(scheme.params.get_margin())

    network_dag = NetworkDAG(scheme.traced)
    network_dag.build_dag()

    for module in net.modules():
        if hasattr(module, "init_orion_params") and callable(module.init_orion_params):
            module.init_orion_params()

    for module in net.modules():
        if hasattr(module, "update_params") and callable(module.update_params):
            module.update_params()

    if scheme.params.get_fuse_modules():
        fuser = Fuser(network_dag)
        fuser.fuse_modules()
        network_dag.remove_fused_batchnorms()

    topo_sort = list(network_dag.topological_sort())

    last_linear = None
    for node in reversed(topo_sort):
        module = network_dag.nodes[node]["module"]
        if isinstance(module, LinearTransform):
            last_linear = node
            break

    print("\nGenerating matrix diagonals...", flush=True)
    for node in topo_sort:
        module = network_dag.nodes[node]["module"]
        if isinstance(module, LinearTransform):
            module.generate_diagonals(last=(node == last_linear))
    checkpoint("after diagonal generation (all layers)", scheme)

    network_dag.find_residuals()

    print("\nRunning bootstrap placement...", flush=True)
    l_eff = len(scheme.params.get_logq()) - 1
    btp_solver = BootstrapSolver(net, network_dag, l_eff=l_eff)
    input_level, num_bootstraps, bootstrapper_slots = btp_solver.solve()
    print(f"Network requires {num_bootstraps} bootstrap ops, "
          f"slots={[int(math.log2(s)) for s in bootstrapper_slots]}", flush=True)

    if bootstrapper_slots:
        for slot_count in bootstrapper_slots:
            scheme.bootstrapper.generate_bootstrapper(slot_count)
    checkpoint("after all bootstrappers generated (boot circuits)", scheme)

    btp_placer = BootstrapPlacer(net, network_dag)
    btp_placer.place_bootstraps()

    print("\nCompiling network layers (per-layer checkpoints)...", flush=True)
    for node in topo_sort:
        node_attrs = network_dag.nodes[node]
        module = node_attrs["module"]
        if isinstance(module, Module):
            module.compile()
            checkpoint(f"after compile: {node} @ level={module.level}", scheme)

    return input_level


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "./configs/resnet_cheddar.yml"
    scheme = orion.init_scheme(config_path)
    trainloader, testloader = get_cifar_datasets(data_dir="./data", batch_size=1)
    net = models.ResNet20()

    net.eval()
    inp, _ = next(iter(testloader))
    out_clear = net(inp)

    orion.fit(net, inp)
    checkpoint("baseline (after fit, before compile)", scheme)

    try:
        input_level = instrumented_compile(scheme, net)
        checkpoint("FINAL (compile complete)", scheme)
        print("\ncompile() completed WITHOUT hitting OOM.")
    except Exception as e:
        print(f"\nSTOPPED with exception: {type(e).__name__}: {e}", flush=True)

    scheme.delete_scheme()
