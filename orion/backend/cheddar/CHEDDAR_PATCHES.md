# Cheddar patches

This backend builds against plain upstream
[Cheddar](https://github.com/scale-snu/cheddar-fhe) plus a small set of
patches, kept here (not in a separate fork/repo) so building this branch
doesn't depend on anything beyond this checkout and upstream Cheddar. Each
patch below touches disjoint files, so they're independent: apply only the
ones you need, in any order.

To apply one: copy its diff block into a file and run, from the root of
your upstream Cheddar checkout (`matmul-encoding-material/cheddar`):

```bash
git apply path/to/the.patch
# or: patch -p1 < path/to/the.patch
```

## Required on `cheddar-backend`

### 1. `managed-memory`

Lets `MemoryPool`'s RMM upstream be swapped to `managed_memory_resource` via
`ORION_CHEDDAR_MANAGED_MEMORY`, so allocations beyond VRAM page to host RAM
instead of OOMing (default `cuda_async` unchanged when unset). A
`statistics_resource_adaptor` exposes `GetPeakDeviceBytes()` for the true
logical peak regardless of physical residency — needed to measure and run
ResNet20's real working set (~41GB peak with `SHARED_BS=64` +
`BOOT_MIN_KS=1`, up to ~71GB with those off) on a 24GB card. See
`README.md`'s "Memory" section for the full env-knob ablation.

```diff
diff --git a/include/core/MemoryPool.h b/include/core/MemoryPool.h
index a9b9093..7f7ea6e 100644
--- a/include/core/MemoryPool.h
+++ b/include/core/MemoryPool.h
@@ -1,18 +1,32 @@
 #pragma once
 
+#include <cstddef>
+#include <memory>
+
 #include <rmm/mr/device/binning_memory_resource.hpp>
-#include <rmm/mr/device/cuda_async_memory_resource.hpp>
-#include <rmm/mr/device/per_device_resource.hpp>
+#include <rmm/mr/device/device_memory_resource.hpp>
+#include <rmm/mr/device/statistics_resource_adaptor.hpp>
 
 #include "core/Parameter.h"
 
 namespace cheddar {
 // After the creation of an MemoryPool object, all memory allocations on the
 // current device uses binning_memory_resouce.
+//
+// The upstream resource is chosen at construction time by the
+// ORION_CHEDDAR_MANAGED_MEMORY env var: unset/"0" keeps the original
+// cuda_async_memory_resource (hard 1:1 GPU-memory cap, the default so other
+// branches rebuilding this library are unaffected); any other value swaps in
+// managed_memory_resource, which lets allocations exceed VRAM by paging
+// overflow to host RAM via CUDA Unified Memory. Both are type-erased behind
+// device_memory_resource so the choice is a runtime one, not a template
+// parameter. A statistics_resource_adaptor sits between the binning pool and
+// its upstream so GetPeakDeviceBytes() can report the true peak footprint
+// regardless of which upstream is active.
 
 class MemoryPool {
-  using DefaultUpstream = rmm::mr::cuda_async_memory_resource;
-  using MemoryPoolBase = rmm::mr::binning_memory_resource<DefaultUpstream>;
+  using Stats = rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>;
+  using MemoryPoolBase = rmm::mr::binning_memory_resource<rmm::mr::device_memory_resource>;
 
  public:
   template <typename word>
@@ -20,8 +34,13 @@ class MemoryPool {
   ~MemoryPool();
 
  private:
-  DefaultUpstream base_;
+  std::unique_ptr<rmm::mr::device_memory_resource> base_;
+  Stats stats_;
   MemoryPoolBase memory_pool_;
 };
 
-}  // namespace cheddar
\ No newline at end of file
+// Peak bytes allocated through the currently-active MemoryPool's statistics
+// adaptor. Returns 0 if no MemoryPool is currently constructed.
+std::size_t GetPeakDeviceBytes();
+
+}  // namespace cheddar
diff --git a/src/core/MemoryPool.cpp b/src/core/MemoryPool.cpp
index 4d8da76..29008bd 100644
--- a/src/core/MemoryPool.cpp
+++ b/src/core/MemoryPool.cpp
@@ -1,10 +1,42 @@
 #include "core/MemoryPool.h"
 
+#include <cstdlib>
+#include <cstring>
+
+#include <rmm/mr/device/cuda_async_memory_resource.hpp>
+#include <rmm/mr/device/managed_memory_resource.hpp>
+#include <rmm/mr/device/per_device_resource.hpp>
+
 namespace cheddar {
 
+namespace {
+
+using StatsAdaptor = rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>;
+
+// Cheddar's Context constructs exactly one MemoryPool for the process's
+// lifetime (mirrors rmm::mr::set_current_device_resource's single-current
+// -resource model), so one global tracking the live adaptor is sufficient.
+StatsAdaptor* g_active_stats = nullptr;
+
+bool ManagedMemoryRequested() {
+  const char* env = std::getenv("ORION_CHEDDAR_MANAGED_MEMORY");
+  return env != nullptr && std::strcmp(env, "0") != 0 && std::strcmp(env, "") != 0;
+}
+
+std::unique_ptr<rmm::mr::device_memory_resource> MakeUpstream() {
+  if (ManagedMemoryRequested()) {
+    return std::make_unique<rmm::mr::managed_memory_resource>();
+  }
+  return std::make_unique<rmm::mr::cuda_async_memory_resource>();
+}
+
+}  // namespace
+
 template <typename word>
 MemoryPool::MemoryPool(const Parameter<word> &param)
-    : base_(), memory_pool_(&base_) {
+    : base_(MakeUpstream()), stats_(base_.get()), memory_pool_(&stats_) {
+  g_active_stats = &stats_;
+
   // Hueristically add bins to save memory and speed-up bootstrapping.
   const int degree = param.degree_;
   const int word_size = param.word_size_;
@@ -37,9 +69,16 @@ MemoryPool::MemoryPool(const Parameter<word> &param)
 MemoryPool::~MemoryPool() {
   // reset to cuda_device_resource
   rmm::mr::set_current_device_resource(nullptr);
+  if (g_active_stats == &stats_) g_active_stats = nullptr;
+}
+
+std::size_t GetPeakDeviceBytes() {
+  return g_active_stats != nullptr
+             ? static_cast<std::size_t>(g_active_stats->get_bytes_counter().peak)
+             : 0;
 }
 
 template MemoryPool::MemoryPool(const Parameter<uint32_t> &param);
 template MemoryPool::MemoryPool(const Parameter<uint64_t> &param);
 
-}  // namespace cheddar
\ No newline at end of file
+}  // namespace cheddar
```

### 2. `evalpoly-argswap`

Fixes a transposed-argument bug in `EvalPolyNode::Compile`:
`EncodeConstant`'s signature is `(constant, level, scale, value)`, and every
call site passes them in that order except this one, which had `scale` and
`level` swapped — the (typically huge) scale value landed in the `int level`
parameter and was later used as a raw vector index, throwing an
out-of-range access.

This is an upstream correctness bug, not an Orion workaround: it fires
whenever the recursive polynomial-split tree bottoms out at a
power-of-two-sized high half (degree=2 hits it directly), independent of
anything the caller controls. Without it, Cheddar's own log-depth
`EvalPoly` (already used internally by `EvalMod`'s bootstrap circuit) is
unusable for any external caller — including the composite sign/ReLU
polynomials this backend evaluates for ResNet's activations.

```diff
diff --git a/src/extension/EvalPoly.cpp b/src/extension/EvalPoly.cpp
index 927b921..0795639 100644
--- a/src/extension/EvalPoly.cpp
+++ b/src/extension/EvalPoly.cpp
@@ -503,8 +503,17 @@ void EvalPolyNode<word>::Compile(ConstContextPtr<word> context,
     auto [_, split_scale] = basis_eval.GetBaseLevelAndScale(split_degree_);
     double high_scale = working_scale / split_scale;
     if (is_high_constant_) {
-      context->encoder_.EncodeConstant(high_constant_, high_scale,
-                                       working_level,
+      // ORION_PATCH (evalpoly-argswap): EncodeConstant's signature is
+      // (constant, level, scale, value) -- every other call site in this
+      // file passes them in that order. This one had scale and level
+      // transposed, so the (typically huge) scale value landed in the
+      // int level parameter, later used as a raw vector index and
+      // throwing an out-of-range access. Upstream bug, not an Orion
+      // workaround: triggers whenever the recursive split tree bottoms
+      // out at a power-of-two-sized high half (degree=2 hits it
+      // directly), independent of anything the caller controls.
+      context->encoder_.EncodeConstant(high_constant_, working_level,
+                                       high_scale,
                                        coefficients_[split_degree_]);
     } else {
       high_->Compile(context, basis_eval, working_level, high_scale, true);
```

### 3. `boot-circuit-eviction`

Adds `BootContext::RemoveEvalSpecialFFT(num_slots)`, which releases a
previously-prepared slot count's CoeffToSlot/SlotToCoeff data, freeing its
device memory (doesn't touch `eval_mod_`, shared across slot counts, or any
evaluation keys). `PrepareEvalSpecialFFT()` must be called again for that
slot count before bootstrapping at it again; no-op if it was never
prepared.

Lets the Orion wrapper (`ORION_CHEDDAR_BOOT_EVICT` in `cheddar_pybind.cu`)
keep at most one boot circuit resident at a time instead of stacking every
slot count a network needs simultaneously — fixes an OOM that surfaces when
`num_stc_levels` is lowered enough to grow each boot circuit's StC BSGS
blocks past the point several simultaneous circuits fit in 24GB.

```diff
diff --git a/include/extension/BootContext.h b/include/extension/BootContext.h
index e419123..917323f 100644
--- a/include/extension/BootContext.h
+++ b/include/extension/BootContext.h
@@ -150,6 +150,23 @@ class BootContext : public Context<word>,
    */
   bool IsBootPrepared(int num_slots) const;
 
+  // ORION_PATCH (boot-circuit-eviction): releases a previously-prepared
+  // num_slots' CoeffToSlot/SlotToCoeff data (eval_fft_/boot_variant_),
+  // freeing its device memory. Does not touch eval_mod_ (shared across all
+  // slot counts) or any evaluation keys (rotation keys live in the
+  // caller's EvkMap, not here). PrepareEvalSpecialFFT() must be called
+  // again for this num_slots before bootstrapping at it again. No-op if
+  // num_slots was never prepared.
+
+  /**
+   * @brief Releases the CoeffToSlot/SlotToCoeff precomputed data for a
+   * given number of slots, freeing its device memory.
+   *
+   * @param num_slots number of slots whose EvalSpecialFFT should be
+   * released
+   */
+  void RemoveEvalSpecialFFT(int num_slots);
+
   /**
    * @brief Performs the trace operation. For s = start_rot_dist, and n =
    * num_accum, res = (input << s) + (input << 2s) + ... + (input << ns).
diff --git a/src/extension/BootContext.cpp b/src/extension/BootContext.cpp
index fcc6b7a..ee825d2 100644
--- a/src/extension/BootContext.cpp
+++ b/src/extension/BootContext.cpp
@@ -39,6 +39,21 @@ BootContext<word>::BootContext(const Parameter<word> &param,
                                const BootParameter &boot_param)
     : Base(param), boot_param_{boot_param} {
   // Check if param and boot_param is consistent
+  //
+  // Considered anchoring this to GetEndLevel() instead (post-StC) so StC
+  // also runs in the reserved zone above the usable chain, the same way
+  // CtS/EvalMod already do -- that would let Boot() restore fully to
+  // default_encryption_level instead of losing num_stc_levels of depth
+  // every time. Measured on ResNet20 (cheddar, LogN=16, num_stc_levels=3):
+  // it works (verified correct), and does cut bootstrap count nearly in
+  // half (75 -> 38). But it's a net latency loss, not a win: the extra
+  // reserved primes widen the RNS representation for every bootstrap's own
+  // CtS/EvalMod/StC, and restoring to a much higher level means every
+  // conv/linear op downstream of a boot runs on a wider (costlier)
+  // ciphertext until the next boot drains it back down. Measured total:
+  // 51.4s (38 boots, full restore) vs 36.0s (75 boots, this GetStCStartLevel()
+  // anchor) -- fewer-but-pricier bootstraps lost to more-but-cheaper ones,
+  // ~30% slower overall. Kept the original anchor for that reason.
   AssertTrue(
       param.max_level_ == boot_param.max_level_ &&
           param.default_encryption_level_ == boot_param.GetStCStartLevel(),
@@ -171,6 +186,12 @@ bool BootContext<word>::IsBootPrepared(int num_slots) const {
          (eval_fft_.find(num_slots) != eval_fft_.end());
 }
 
+template <typename word>
+void BootContext<word>::RemoveEvalSpecialFFT(int num_slots) {
+  eval_fft_.erase(num_slots);
+  boot_variant_.erase(num_slots);
+}
+
 template <typename word>
 void BootContext<word>::AddRequiredRotations(EvkRequest &req, int num_slots,
                                              bool min_ks) const {
```

### 4. `fused-dispatch-beta1`

Fixes a one-element dispatch gap in `HoistHandler`'s two fused kernels.
Both `BSFusedKeyMult` and `GSFusedPAccum` pick a compile-time-padded
kernel instantiation with `constexpr_for<1, ...>`, i.e. starting at
padded size 2 — so `num_accum == 1` (a single decomposition digit,
beta = 1) and `num_bs == 1` (a single baby-step index) match no
instantiation and the kernel silently never launches, leaving the
output zero. No error, no assert: the accumulation just doesn't happen.

Upstream bug, not an Orion workaround — but upstream's own bootstrap
never trips it, because boot circuits run at high levels where the digit
count is always ≥ 2. It surfaces the moment `HoistHandler` is used where
it's most attractive: hoisted (double-hoisted) linear transforms
mid-network at low levels, where `num_q ≤ alpha` gives beta = 1. Without
this patch the `ORION_CHEDDAR_DOUBLE_HOIST` path in `bindings.py`
produces wrong results (giant-step term only, baby contributions zero)
at low levels; with it, all oracle LT tests pass at ~3e-8.

The fix starts the dispatch at padded size 1 (`constexpr_for<0, ...>`)
and guards the lower-bound check, which is ill-formed for `i == 0`.

```diff
diff --git a/src/extension/Hoist.cu b/src/extension/Hoist.cu
index f561b66..35f6d3e 100644
--- a/src/extension/Hoist.cu
+++ b/src/extension/Hoist.cu
@@ -324,10 +324,20 @@ void HoistHandler<word>::BSFusedKeyMult(
   AssertTrue(num_accum <= (1 << max_log_beta_),
              "num_accum should not be greater than " +
                  std::to_string(1 << max_log_beta_));
-  constexpr_for<1, max_log_beta_ + 1>([&](auto i) {
+  // ORION_PATCH (fused-dispatch-beta1): the dispatch used to start at
+  // i=1, so num_accum == 1 (beta == 1: low-level inputs, num_q <= alpha)
+  // matched no instantiation and the kernel silently never launched --
+  // every baby-step rotation came back zero. Bootstrap never hits this
+  // (its circuits run at high levels where beta >= 2), so it only
+  // surfaces when using the hoisted path at low levels, e.g. hoisted
+  // linear transforms mid-network. Start at i=0 (padded size 1) and
+  // guard the lower-bound check, which is ill-formed for i == 0.
+  constexpr_for<0, max_log_beta_ + 1>([&](auto i) {
     constexpr int num_accum_padded = 1 << i;
     if (num_accum > num_accum_padded) return;
-    if (num_accum <= (1 << (i - 1))) return;
+    if constexpr (i > 0) {
+      if (num_accum <= (1 << (i - 1))) return;
+    }
     kernel::BSFusedKernel<word, num_accum_padded><<<grid_dim, block_dim>>>(
         dst_b_d_ptrs.data(), dst_a_d_ptrs.data(), modup_d_ptrs.data(),
         key_b_d_ptrs.data(), key_a_d_ptrs.data(), num_accum, num_rotations,
@@ -436,10 +446,15 @@ void HoistHandler<word>::GSFusedPAccum(ConstContextPtr<word> context,
   dim3 block_dim(kernel_block_dim_);
   dim3 grid_dim(num_primes * context->param_.degree_ / kernel_block_dim_);
 
-  constexpr_for<1, max_log_bs_ + 1>([&](auto i) {
+  // ORION_PATCH (fused-dispatch-beta1): same one-element dispatch gap as
+  // BSFusedKeyMult above -- num_bs == 1 (a single baby-step index)
+  // matched no instantiation and the accumulation silently stayed zero.
+  constexpr_for<0, max_log_bs_ + 1>([&](auto i) {
     constexpr int num_bs_padded = 1 << i;
     if (num_bs > num_bs_padded) return;
-    if (num_bs <= (1 << (i - 1))) return;
+    if constexpr (i > 0) {
+      if (num_bs <= (1 << (i - 1))) return;
+    }
     kernel::GSFusedKernel<word, num_bs_padded><<<grid_dim, block_dim>>>(
         dst_b_d_ptrs.data(), dst_a_d_ptrs.data(), bx_d_ptrs.data(),
         ax_d_ptrs.data(), mx_d_ptrs.data(), num_bs, num_gs, primes, inv_primes);
```

### 5. `seeded-ciphertext`

Required for `bench_sweep.py --seeded-upload` (and nothing else -- it is
inert unless those bindings are called).

A fresh RLWE ciphertext is `(b, a)` with `a` uniform in R_q and
`b = -a*s + m + e`. Because `a` is uniform and independent of both the
message and the secret, a client can derive it from a seed and transmit
only `(b, seed)`, letting the server rebuild `a`. That halves upload
bytes for every fresh ciphertext. (It is upload-only: after any
homomorphic operation `a` is no longer seed-derived, so results coming
back cannot be compressed this way.)

Upstream samples `a` inside `UserInterface::Encrypt` through a static
generator with no way to seed it, and `SampleRandomPolynomial` is
private. This patch adds:

- `Random::SetSeed(uint64_t)` to reseed that generator, with default
  ELF visibility on the class so Orion's `-fvisibility=hidden`
  extension shares the one generator rather than linking a private copy
  (identical to the `keygen-seed` patch used by the multi-GPU work --
  apply either one, not both).
- `UserInterface::ExpandSeededAx(Ct&, uint64_t)`, a public entry point
  that reseeds and resamples a ciphertext's `ax_` in place.

Both make key generation and error sampling reproducible when used,
which is acceptable only because `UserInterface` is already documented
upstream as test-only and not a secure implementation. A deployment
would expand `a` from a CSPRNG (ideally on device -- see the runtime
note below), not from `std::mt19937`.

**Security note.** Seeds must be fresh per ciphertext. Reusing one `a`
across two messages under the same key gives
`b1 - b2 = m1 - m2 + e1 - e2`, leaking the message difference outright.

**Runtime note.** Upstream `SampleRandomPolynomial` generates on the
host with `std::uniform_int_distribution` and then copies to device, so
expansion costs ~6.5 ms per ciphertext at LogN=16 -- fine for a handful
of ciphertexts, but it dominates at high selector widths (+557 ms at
threshold 5,000,000, where 85 ciphertexts are uploaded). A device-side
CSPRNG would remove essentially all of that; the measured numbers in
`examples/criteo/PAPER_NUMBERS.md` should be read as an upper bound on
the cost of this technique, not its floor.

```diff
diff --git a/include/Random.h b/include/Random.h
index eb0cc3d..7b905c7 100644
--- a/include/Random.h
+++ b/include/Random.h
@@ -2,6 +2,7 @@
 
 #include <algorithm>
 #include <cmath>
+#include <cstdint>
 #include <iostream>
 #include <random>
 
@@ -15,7 +16,18 @@ namespace cheddar {
  * production.
  *
  */
+// Default visibility is deliberate. gen_ below is a static inline
+// member, i.e. a STB_GNU_UNIQUE symbol that the dynamic linker unifies
+// across shared objects -- but only if every object exports it. Orion's
+// pybind extension is built -fvisibility=hidden, so without this
+// attribute it links its own private copy and SetSeed() would reseed a
+// generator that nothing samples from, silently leaving shards with
+// different secrets.
+#if defined(__GNUC__)
+class __attribute__((visibility("default"))) Random {
+#else
 class Random {
+#endif
  public:
   Random() = default;
   Random(const Random &) = delete;
@@ -58,6 +70,25 @@ class Random {
     });
   }
 
+  /**
+   * @brief Reseed the shared generator deterministically.
+   *
+   * Multi-GPU shards run as separate processes and must share one key
+   * space (see examples/hydia/MULTI-GPU.md): secrets and evaluation
+   * keys are sampled through this generator, so seeding every shard
+   * identically before UserInterface construction makes them generate
+   * the same keys without any key material crossing the process
+   * boundary. Leaving it unseeded keeps the previous
+   * nondeterministic behaviour.
+   *
+   * This makes key generation fully reproducible, error sampling
+   * included, which is acceptable only because UserInterface is
+   * already documented as test-only and not a secure implementation.
+   *
+   * @param seed the seed value
+   */
+  static void SetSeed(uint64_t seed) { gen_.seed(seed); }
+
   template <typename T>
   static void SampleDiscreteNormal(T *ptr, int length, double mean,
                                    double stddev, double max_sigma = 6) {
diff --git a/include/UserInterface.h b/include/UserInterface.h
index 2ff294d..9d2728e 100644
--- a/include/UserInterface.h
+++ b/include/UserInterface.h
@@ -78,6 +78,27 @@ class UserInterface {
    */
   void PrepareRotationKey(int rot_idx, int max_level = -1);
 
+  /**
+   * @brief Regenerate a ciphertext's `a` polynomial from a seed.
+   *
+   * A fresh RLWE ciphertext is (b, a) with a uniform in R_q and
+   * b = -a*s + m + e. Because a is uniform and independent of both the
+   * message and the secret, a client may derive it from a seed and
+   * transmit only (b, seed); the server calls this to rebuild a and
+   * recover the full ciphertext. Halves upload size.
+   *
+   * The seed MUST be fresh per ciphertext: reusing one a across two
+   * messages under the same key leaks m1-m2 directly from b1-b2.
+   *
+   * Reseeds the shared generator, so any sampling that follows is also
+   * deterministic until reseeded -- acceptable only because
+   * UserInterface is already documented as test-only.
+   *
+   * @param ctxt ciphertext whose ax_ is overwritten
+   * @param seed the seed the client used
+   */
+  void ExpandSeededAx(Ct &ctxt, uint64_t seed) const;
+
   /**
    * @brief Prepare rotation keys for the given EvkRequest.
    *
diff --git a/src/UserInterface.cu b/src/UserInterface.cu
index 7b6f79b..c091f11 100644
--- a/src/UserInterface.cu
+++ b/src/UserInterface.cu
@@ -459,6 +459,12 @@ void UserInterface<word>::PrepareEvk(int key_idx, const NPInfo &np,
 
 // We can regard this random polynomial as having any form we want
 // It can be regarded to be NTT-applied or not (with or without Montgomery form)
+template <typename word>
+void UserInterface<word>::ExpandSeededAx(Ct &ctxt, uint64_t seed) const {
+  Random::SetSeed(seed);
+  SampleRandomPolynomial(ctxt.ax_, ctxt.GetNP());
+}
+
 template <typename word>
 void UserInterface<word>::SampleRandomPolynomial(Dv &poly,
                                                  const NPInfo &np) const {
```

## Not used on `cheddar-backend`

### 6. `fused-rotate`

Makes `HoistHandler::BSFusedKeyMult` public and static (plus adds
`PopulateCM`) so an external caller can run a fused hoisted N-way rotation
— KeyMult + MAC + Automorphism in one kernel launch — without constructing
a `HoistHandler`. Powers `hoist_mode="fused"` in `bindings.py`.

**This branch's `bindings.py` only implements `hoist_mode` `none`/`single`
— `"fused"` isn't wired in here, so this patch isn't needed to build or run
`cheddar-backend`.** It's required on `feat/cheddar-32bit`, which does wire
up `hoist_mode="fused"`. Included here for completeness / in case this
branch picks up fused-rotate support later.

```diff
diff --git a/include/extension/Hoist.h b/include/extension/Hoist.h
index a270294..0444b9f 100644
--- a/include/extension/Hoist.h
+++ b/include/extension/Hoist.h
@@ -59,10 +59,30 @@ class HoistHandler {
   void GSFusedPAccum(ConstContextPtr<word> context, std::map<int, Ct> &results,
                      const std::vector<int> &gs_indices,
                      const std::map<int, Ct> &bs) const;
-  void BSFusedKeyMult(ConstContextPtr<word> context, std::map<int, Ct> &res,
-                      std::vector<Dv> &a_modup, const Ct &a_orig,
-                      const EvkMap<word> &keys, std::vector<int> &rotations,
-                      const Dv &input_bx_pseudo_modup) const;
+
+ public:
+  // ORION_PATCH (fused-rotate): BSFusedKeyMult made public + static so the
+  // Orion backend can run a fused hoisted N-way rotation (KeyMult + MAC +
+  // Automorphism in one kernel) without constructing a HoistHandler. The
+  // method only reads static class constants (kernel_block_dim_,
+  // max_log_beta_) and its arguments -- no instance state -- so static is
+  // safe. Outputs land in res permuted, in the extended basis; callers
+  // ModDown each entry.
+  static void BSFusedKeyMult(ConstContextPtr<word> context,
+                             std::map<int, Ct> &res, std::vector<Dv> &a_modup,
+                             const Ct &a_orig, const EvkMap<word> &keys,
+                             std::vector<int> &rotations,
+                             const Dv &input_bx_pseudo_modup);
+
+  // ORION_PATCH (fused-rotate): the fused kernel reads __constant__
+  // tables that the HoistHandler constructor normally populates; those
+  // symbols are per-TU, so external callers of BSFusedKeyMult must
+  // populate Hoist.cu's copy through this entry point (defined in
+  // Hoist.cu). Repopulates unconditionally: schemes can be rebuilt with
+  // different parameters within one process.
+  static void PopulateCM(const Parameter<word> &param);
+
+ private:
 
   // evaluation-related methods
   void EvaluateSingleAccum(ConstContextPtr<word> context, Ct &res,
diff --git a/src/extension/Hoist.cu b/src/extension/Hoist.cu
index 0e09bb3..f561b66 100644
--- a/src/extension/Hoist.cu
+++ b/src/extension/Hoist.cu
@@ -204,11 +204,21 @@ std::pair<int, int> HoistHandler<word>::CheckStrideMinKS() const {
   return {bs_gcd, gs_gcd};
 }
 
+// ORION_PATCH (fused-rotate): populate THIS translation unit's constant
+// memory (see header comment).
+template <typename word>
+void HoistHandler<word>::PopulateCM(const Parameter<word> &param) {
+  PopulateConstantMemory(param);
+  cm_populated_ = true;
+}
+
+// ORION_PATCH (fused-rotate): definition matches the now-static public
+// declaration (trailing const dropped; static is implied out-of-class).
 template <typename word>
 void HoistHandler<word>::BSFusedKeyMult(
     ConstContextPtr<word> context, std::map<int, Ct> &res,
     std::vector<Dv> &a_modup, const Ct &a_orig, const EvkMap<word> &keys,
-    std::vector<int> &rotations, const Dv &input_bx_pseudo_modup) const {
+    std::vector<int> &rotations, const Dv &input_bx_pseudo_modup) {
   NPInfo a_orig_np = a_orig.GetNP();
   int level = context->param_.NPToLevel(a_orig_np);
   int num_main = a_orig_np.num_main_;
```
