# Cheddar backend

Wraps [Cheddar](https://github.com/scale-snu/cheddar-fhe) (SNU SCALE lab,
C++/CUDA CKKS) behind the same ID-based interface as the desilo backend.
64-bit word mode.

Everything the general interface exercises works: encode/decode,
encrypt/decrypt, ct-ct/ct-pt/scalar arithmetic, rescale, rotation
(incl. hoisted batch rotation), linear transform, polynomial evaluation,
and bootstrap. `run_lola`/`run_mlp` run end to end;
`tests/oracle/ --backend=cheddar` is 71/71, plus the `slow`-marked
bootstrap tests pass (`pytest -m slow -k cheddar tests/oracle/`).
(`run_resnet`/`run_helrm` still need a `configs/resnet_cheddar.yml`,
which doesn't exist yet.)

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

### Bootstrap quirks

- **Reserved chain levels (level model).** `BootContext::Create` asserts
  `default_encryption_level == max_level - num_cts_levels -
  GetNumEvalModLevels()`: the topmost `num_cts_levels + eval_mod_levels`
  primes of the modulus chain are reserved for the boot circuit
  (CoeffToSlot + EvalMod) and sit *above* the usable chain. So the LogQ
  in a config is the *usable* level range; `setup_scheme` appends the
  reserved boot primes natively (their count comes from
  `BootNumEvalModLevels()`, a fixed `Log2Ceil(31) + 3 = 8` baked into the
  vendored `BootParameter`). This mirrors how lattigo silently extends
  its own chain for bootstrap; unlike naive expectation, the LogQ you
  write is *not* the full chain the card allocates.

- **Single context, not two.** `BootContext` derives from `Context`, so
  when boot params are present `setup_scheme` builds the `BootContext`
  and uses it as *the* scheme context (regular ops go through it
  unchanged). Keeping a separate regular `Context` + `BootContext`
  doubles resident memory and OOMs full-slot LogN=16 boot on a 24GB card
  (~24GB vs ~15GB for upstream's own single-context `boot_test`). Non-boot
  configs keep the plain `Context` path.

- **Minimum 256 slots.** Cheddar's `EvalSpecialFFT` asserts
  `num_slots >= 256`, so sparse bootstrap below 256 slots is unsupported
  (lattigo/desilo go lower).

- **Input-scale snapping (precision, has a tradeoff worth knowing).**
  Cheddar's `Boot` precomputes its EvalMod/CtS/StC constants *once* from
  the scheme's fixed `base_scale`, i.e. it *assumes the input ciphertext
  is at exactly `base_scale`.* This is a deliberate precompute-once design
  (its own unittest feeds boot a fresh level-0 encode, which is exactly
  `base_scale`), not an oversight -- lattigo instead reads the input's
  actual scale and adjusts per-call.

  But cheddar rescales by the *exact* prime (not a fixed target), so the
  scale drifts from `base_scale`; and because `MulScalar`-then-`Rescale`
  squares the scale, a ~2^-20 per-prime offset *compounds*, reaching ~1%
  after a dozen rescales. Bootstrap precision is empirically capped near
  `-log2(|scale/base_scale - 1|)` (measured: exact -> 19 bits,
  2^-12 -> 13, 2^-10 -> 11, 2^-6.7 (1%) -> 7.7). So a drained ciphertext
  bootstraps at only ~8 bits despite Boot itself being ~19-bit accurate
  on a clean input.

  The `Bootstrap` binding fixes this by *snapping*: relabel the input's
  scale to exactly `base_scale` (a pure `SetScale`, no data change), which
  makes Boot's assumption true but rescales the *represented* message by
  `r = scale/base_scale`; boot the snapped copy; then divide the result
  by `r` (a `MulScalarFloat` + `Rescale`) to recover the true message.
  Restores full ~19-bit precision.

  **Tradeoff / overhead.** The `1/r` correction costs **one level** per
  bootstrap. That's the price of producing a *clean* `base_scale` output.
  Two cheaper alternatives exist but weren't taken:
  - *Relabel the output* to `base_scale * r` (= the input's original
    drifted scale) instead of multiplying -- zero level cost, but leaves
    the boot output at a ~1% drifted scale. The input already carried that
    drift and Orion tolerates it elsewhere, so this is *likely* safe, but
    a downstream `ct+ct` add combining the boot output with a ~0%-drift
    operand could trip the `kDriftTolerance` (1e-4) check. Circuit-
    dependent; the level-consuming clean output was chosen as the
    conservative default.
  - *Eliminate the drift at the source* via fixed-scale rescaling (relabel
    to exactly `base_scale` after every rescale, absorbing the error into
    noise -- lattigo's default). Then boot always sees `base_scale`, no
    snap and no level cost ever, but it touches every rescale, not just
    boot.

  A fresh, undrifted input (`r == 1`, e.g. bootstrapping a bare encode)
  skips the whole dance.
