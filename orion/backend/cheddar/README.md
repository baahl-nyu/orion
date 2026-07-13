# Cheddar backend

Wraps [Cheddar](https://github.com/scale-snu/cheddar-fhe) (SNU SCALE lab,
C++/CUDA CKKS) behind the same ID-based interface as the desilo backend.
64-bit word mode.

Not yet implemented: bootstrap (raises `NotImplementedError`) -- so
`run_resnet`/`run_helrm` and the bootstrap section of `tests/oracle/`
don't work on this backend yet. Everything else does: encode/decode,
encrypt/decrypt, ct-ct/ct-pt/scalar arithmetic, rescale, rotation
(incl. hoisted batch rotation), linear transform, and polynomial
evaluation -- `run_lola`/`run_mlp` run end to end
(`tests/oracle/ --backend=cheddar` is 71/71 outside the `slow`-marked
bootstrap tests).

## Build

1. Build Cheddar itself (once). Plain upstream, no patches needed:

```bash
git clone https://github.com/scale-snu/cheddar-fhe.git \
    matmul-encoding-material/cheddar
cd matmul-encoding-material/cheddar
CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release -DUSE_GMP=ON \
    -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11
cmake --build build -j
```

Requires CMake >= 3.24 (`pip install cmake` if the system one is older)
and network access on first configure (rmm via FetchContent). The
explicit gcc-11 pins work around CMake's CUDA compiler probe grabbing
mixed gcc-9/gcc-11 headers on this box (same fix as HEonGPU).

2. Build the pybind11 extension:

```bash
cd orion/backend/cheddar/ext
CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 cmake -S . -B build \
    -D CHEDDAR_SOURCE_DIR=$PWD/../../../../matmul-encoding-material/cheddar \
    -D CMAKE_CUDA_ARCHITECTURES=86 \
    -D CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11 \
    -D Python3_EXECUTABLE=$(which python)
cmake --build build -j 8
# Output: orion/backend/cheddar/_cheddar_native.cpython-310-x86_64-linux-gnu.so
```

The extension links `libcheddar.so` from `CHEDDAR_BUILD_DIR` (defaults
to `<CHEDDAR_SOURCE_DIR>/build`) with an rpath, so no LD_LIBRARY_PATH
is needed at runtime.

## Use

```yaml
orion:
  backend: cheddar
  device: gpu        # GPU-only backend
```

Examples:

```bash
python examples/run_lola.py configs/lola_cheddar.yml
python examples/run_mlp.py configs/mlp_cheddar.yml
```

Oracle tests:

```bash
pytest tests/oracle/ --backend=cheddar
```

Env knobs:

- `ORION_CHEDDAR_HOIST_MODE`: `none` (per-shift rotation loop) or
  `single` (shared ModUp per batch, the default). Both build against
  plain upstream Cheddar.

## Quirks discovered

- `UserInterface::PrepareRotationKey`: always pass an explicit
  `max_level` (the wrapper passes the scheme max). The docstring's
  "`-1` → param max" default is stale — in the implementation `-1` is
  the internal sentinel for the dense-to-sparse (bootstrap) key shape
  (`GetNPForEvk(-1)`), so relying on the default yields a beta-1 key
  and "Beta mismatch" at rotation time. Upstream's own tests and the
  `EvkRequest` path always pass explicit levels; explicit level is the
  intended usage.
- Cheddar's `AssertTrue` calls `std::exit`, not throw. The wrapper
  pre-checks every level/scale condition and raises Python exceptions
  instead, so misuse can't kill the pytest process.
- Rotation keys are per-distance, normalized to [0, slots); rotation by
  0 is served as a plain copy (kernels use `rot_batch(ct, [0])` as a
  cheap clone).
