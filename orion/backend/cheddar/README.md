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

- `run_resnet.py configs/resnet_cheddar.yml` (ResNet20/CIFAR-10) runs end
  to end: MAE 0.0008, ~36s on a 24GB card, with
  `ORION_CHEDDAR_POW2_ROTATE=1 ORION_CHEDDAR_MANAGED_MEMORY=1` (needed --
  ResNet20's exact-key working set is ~90GB; see "Env knobs" below).
- `examples/criteo/run_orion.py examples/criteo/params_cheddar.yaml`
  (Criteo HE-LRM) is verified against a synthetic (untrained) instance of
  the real architecture at `batch_size=16` (MAE 0.0004; 16 is a floor, not
  a tuning choice -- HELRM's narrowest bootstrapped layer is 16 wide, and
  Cheddar's `EvalSpecialFFT` hard-asserts `num_slots >= 256`). Needs the
  same mitigations as ResNet20 plus `ORION_CHEDDAR_BOOT_MIN_KS=1`. Not yet
  runnable through `run_orion.py`'s real entrypoint, which reads a trained
  checkpoint (`model/model.pt`) that requires the actual Criteo dataset
  and training run, neither of which has happened yet.

## Build

1. Build Cheddar itself (once). Clone plain upstream, then apply this
   branch's 3 required patches from
   [CHEDDAR_PATCHES.md](CHEDDAR_PATCHES.md) (`managed-memory`,
   `evalpoly-argswap`, `boot-circuit-eviction` -- copy each diff block into
   a file and `git apply` it; a 4th patch, `fused-rotate`, is documented
   there too but isn't needed on this branch):

```bash
git clone https://github.com/scale-snu/cheddar-fhe.git \
    matmul-encoding-material/cheddar
cd matmul-encoding-material/cheddar
git apply /path/to/managed-memory.patch
git apply /path/to/evalpoly-argswap.patch
git apply /path/to/boot-circuit-eviction.patch
CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release -DUSE_GMP=ON \
    -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11
cmake --build build -j
```

Without these patches: native `EvalPoly` (polynomial degree >= 2, used for
ResNet's composite ReLU/sign activation) crashes with an out-of-range
access, and `RemoveEvalSpecialFFT`/`GetPeakDeviceMemoryMB` won't exist for
the extension to link against -- see [CHEDDAR_PATCHES.md](CHEDDAR_PATCHES.md)
for why each one is needed and what it does.

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
ORION_CHEDDAR_POW2_ROTATE=1 ORION_CHEDDAR_MANAGED_MEMORY=1 \
    python examples/run_resnet.py configs/resnet_cheddar.yml
```

Oracle tests:

```bash
pytest tests/oracle/ --backend=cheddar
```

Env knobs:

- `ORION_CHEDDAR_HOIST_MODE` (read in `orion/core/orion.py`): `none`
  (per-shift rotation loop) or `single` (shared ModUp per batch, the
  default). Both work on plain upstream Cheddar; `single` is default.
- `ORION_CHEDDAR_POW2_ROTATE=1`: swap per-layer exact-BSGS-distance
  rotation keys for a fixed universal power-of-two key set (~log2(slots)
  keys total, shared by every layer). Cuts ResNet20's key memory from
  ~90GB to ~15GB at the cost of up to log2(slots) rotations per logical
  shift instead of 1. Off (exact-key) by default.
- `ORION_CHEDDAR_MANAGED_MEMORY=1` (read inside the vendored C++
  `MemoryPool`, requires the `managed-memory` patch): page allocations
  beyond VRAM to host RAM via CUDA managed memory instead of OOMing.
  cuda_async (hard VRAM cap) by default.
- `ORION_CHEDDAR_BOOT_MIN_KS=1`: generate the minimum rotation-key set
  for bootstrap instead of the full set -- less memory, slower Boot. Off
  by default.
- `ORION_CHEDDAR_BOOT_EVICT=1` (requires the `boot-circuit-eviction`
  patch): keep at most one boot circuit's CoeffToSlot/SlotToCoeff data
  resident at a time, evicting others before preparing a new one, instead
  of keeping every slot count a network needs simultaneously resident.
  Off by default.

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

- **Aggregate key memory on big models (`min_ks`).** A deep model like
  ResNet bootstraps at several slot counts (32768, 16384, 8192, 4096 as
  the spatial dims shrink), and Orion prepares a *separate* boot circuit
  per slot count -- all resident at once, each with its own rotation
  keys, on top of every linear-transform's rotation keys. With the
  default full key set (`min_ks=false`) that overflows a 24GB card (OOM
  while generating the conv rotation keys, *after* the boot circuits are
  built). Set `ORION_CHEDDAR_BOOT_MIN_KS=1` to generate the minimum boot
  key set instead -- far less memory, slower Boot. Unlike lattigo/desilo,
  cheddar has no key/diagonal serialization (`GenerateAndSerializeRotationKey`
  / `LoadRotationKey` raise `NotImplementedError`), so `io_mode: load/save`
  can't stream keys from disk -- everything must fit in GPU memory at once.

- **`GetModuliChain` returns the real usable primes.** The production
  bootstrap path (and batch norm / extract / embedding layers) call
  `get_moduli_chain()[level]` to encode plaintexts at `scale = q[level]`
  for errorless rescaling. Desilo returns a `default_scale` stub (it
  manages scale internally); cheddar returns the *actual* usable-level
  primes (excluding the reserved boot primes), since it rescales by the
  exact prime so the real value is what keeps rescaling clean. The oracle
  bootstrap tests don't exercise this (they encode the prescale at the
  default scale), so it only surfaced running a real model.

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
