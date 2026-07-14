"""Cheddar CKKS backend.

Mirrors the same ID-based interface as DeSiLoLibrary / LattigoLibrary so
that orion/backend/python/ keeps working unchanged. All actual FHE work
goes through the native ``_cheddar_native`` pybind11 module built from
``orion/backend/cheddar/ext/``.

Bootstrap orchestrates Cheddar's native BootContext (CoeffToSlot / EvalMod /
SlotToCoeff) from this layer -- see NewBootstrapper/Bootstrap below and
extension/BootContext.h. Linear transform and polynomial evaluation are
implemented entirely at this layer (rotate + ct-pt multiply + add for LT;
Horner's method over ct-ct multiply for polynomials) rather than via
native evaluator objects -- see GenerateLinearTransform and
EvaluatePolynomial.

Word size: uint64. Cheddar's Parameter takes explicit prime lists rather
than bit sizes, so this module converts Orion's LogQ/LogP into
NTT-friendly primes (p = 1 mod 2N) before calling setup_scheme.
"""

from __future__ import annotations

import atexit
import math
import os
from typing import Sequence

import numpy as np

try:
    from . import _cheddar_native as _native
except ImportError as e:  # pragma: no cover - import-time diagnostic
    raise ImportError(
        "Could not import _cheddar_native. Build the extension via the "
        "CMake steps in orion/backend/cheddar/README.md. "
        f"Underlying error: {e}"
    ) from None


# Process-exit cleanup, same rationale as the HEonGPU backend: the
# global BackendState's GPU allocations must destruct while the CUDA
# driver context is still alive, not during interpreter teardown.
_atexit_registered = False


def _register_atexit_cleanup() -> None:
    global _atexit_registered
    if _atexit_registered:
        return

    def _cleanup() -> None:
        try:
            _native.delete_scheme()
        except Exception:
            pass

    atexit.register(_cleanup)
    _atexit_registered = True


# ---------------------------------------------------------------------------
# Prime generation (64-bit): NTT-friendly primes p = 1 (mod 2N), chosen
# closest to 2^bits alternating above/below (lattigo-style) so per-level
# scale drift stays balanced. Deterministic Miller-Rabin is exact for
# all 64-bit inputs with the standard 12-witness set.
# ---------------------------------------------------------------------------

_MR_WITNESSES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in _MR_WITNESSES:
        if n % p == 0:
            return n == p
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MR_WITNESSES:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _gen_primes(bit_sizes: Sequence[int], logn: int, used: set[int]) -> list[int]:
    """One prime per requested bit size, p = 1 mod 2^(logn+1), each as
    close to 2^bits as possible, alternating above/below per repeated
    bit size, all distinct across the whole scheme (tracked in `used`)."""
    m = 1 << (logn + 1)  # primes must be = 1 mod 2N for the NTT
    out: list[int] = []
    for bits in bit_sizes:
        if not 2 < bits < 63:
            raise ValueError(f"prime bit size {bits} out of range for uint64")
        target = 1 << bits
        # Candidates walk away from 2^bits in steps of m: up (2^bits+1+k*m)
        # and down (2^bits+1-k*m), nearest first.
        k = 0
        found = None
        while found is None:
            k += 1
            for cand in (target + 1 - k * m, target + 1 + k * m):
                if cand in used or cand < 3 or cand >= (1 << 63):
                    continue
                if _is_prime(cand):
                    found = cand
                    break
        used.add(found)
        out.append(found)
    return out


class CheddarLibrary:
    """Cheddar CKKS backend wrapping ``_cheddar_native``.

    ``setup_bindings(orion_params)`` is the single entry point
    Scheme.setup_backend uses. All other methods are called per-op from
    Context / kernel code via the ID handles.

    Args:
        hoist_mode: "none" (per-shift rotation loop) or "single"
            (wrapper-level single hoisting: shares one ModUp across a
            batch of rotations; builds against plain upstream Cheddar,
            no patches needed).
    """

    _HOIST_MODE_BY_NAME = {"none": 0, "single": 1}

    def __init__(self, *, hoist_mode: str = "none"):
        if hoist_mode not in self._HOIST_MODE_BY_NAME:
            raise ValueError(
                f"hoist_mode must be one of "
                f"{list(self._HOIST_MODE_BY_NAME.keys())}, got {hoist_mode!r}"
            )
        self._hoist_mode = self._HOIST_MODE_BY_NAME[hoist_mode]
        self._hoist_mode_name = hoist_mode

        self._default_scale: int | None = None
        self._max_level: int | None = None
        self._slots: int | None = None
        self._device: str | None = None
        # Usable-level moduli (one prime per Orion level), set in setup_scheme.
        self._usable_primes: list[int] = []
        # Generation this instance owns; DeleteScheme only tears down
        # native state if it is still the live generation (the bench
        # runner GC-finalizes the OLD scheme after the NEW one is set
        # up -- an unconditional delete would wipe the new state).
        self._generation: int | None = None

        # Linear-transform state. No native LT object -- evaluated at this
        # layer via rotate + ct-pt multiply + add (optionally BSGS-batched),
        # same approach as DeSiLoLibrary.
        self._transforms: dict[int, dict] = {}
        self._next_lt_id = 1

        # Polynomial state. No native evaluator -- coefficients are stored
        # here and evaluated via Horner's method over ct-ct/ct-scalar ops
        # (see EvaluatePolynomial).
        self._polynomials: dict[int, tuple[list[float], str]] = {}
        self._next_poly_id = 1

        _register_atexit_cleanup()

    # ------------------------------------------------------------------
    # setup_bindings  (called by Scheme.setup_backend)
    # ------------------------------------------------------------------

    def setup_bindings(self, orion_params) -> None:
        self.setup_scheme(orion_params)
        self.setup_tensor_binds()
        self.setup_key_generator()
        self.setup_encoder()
        self.setup_encryptor()
        self.setup_evaluator()
        self.setup_poly_evaluator()
        self.setup_lt_evaluator()
        self.setup_bootstrapper()

    def setup_scheme(self, orion_params) -> None:
        logq = orion_params.get_logq()
        logp = orion_params.get_logp()
        logscale = orion_params.get_logscale()
        logn = orion_params.get_logn()

        self._max_level = len(logq) - 1
        self._default_scale = 1 << logscale
        self._slots = orion_params.get_slots()
        self._device = orion_params.get_device()

        if self._device != "gpu":
            raise RuntimeError("Cheddar backend only supports device='gpu'.")

        # boot_params.LogP is a lattigo-specific knob (extra auxiliary
        # primes for its bootstrap circuit) -- ignored here, same as
        # desilo. Cheddar's BootContext reuses this scheme's own aux
        # primes for its internal key-switching; the knobs it actually
        # needs (num_cts_levels/num_stc_levels/log_message_ratio) come
        # from boot_params separately.
        #
        # Bootstrap is enabled iff num_cts_levels *and* num_stc_levels are
        # explicitly given in boot_params. When enabled, the topmost
        # (num_cts_levels + eval_mod_levels) primes of the chain are
        # reserved for the boot circuit (CoeffToSlot + EvalMod) and are
        # NOT part of Orion's usable level range -- BootContext asserts
        # default_encryption_level == max_level - num_cts_levels -
        # GetNumEvalModLevels(). This mirrors how lattigo silently extends
        # its modulus chain for the bootstrap circuit: Orion's LogQ is the
        # usable chain; the reserved primes sit above it.
        raw_cts = orion_params.get_boot_num_cts_levels()
        raw_stc = orion_params.get_boot_num_stc_levels()
        self._boot_enabled = raw_cts is not None and raw_stc is not None
        self._boot_num_cts_levels = raw_cts or 4
        self._boot_num_stc_levels = raw_stc or 3
        self._boot_log_message_ratio = (
            orion_params.get_boot_log_message_ratio() or 5)

        used: set[int] = set()
        main_primes = _gen_primes(logq, logn, used)
        default_enc_level = len(main_primes) - 1  # usable-chain top
        # The usable-level moduli, indexed by Orion level (0..default_enc_level).
        # GetModuliChain returns these -- callers encode plaintexts at
        # scale = q[level] for errorless rescaling. Excludes the reserved
        # boot primes appended below (those aren't Orion-visible levels).
        self._usable_primes = list(main_primes)

        if self._boot_enabled:
            eval_mod_levels = _native.BootNumEvalModLevels()
            num_reserved = self._boot_num_cts_levels + eval_mod_levels
            # Reserved boot-circuit primes. Sized like Cheddar's own 64-bit
            # reference set (parameters/bootparam_40_64bit.json), whose boot
            # levels run ~15 bits above the scale (scale 2^40 -> ~2^55
            # primes) to give EvalMod/CtS/StC precision headroom. Capped at
            # 61 to stay comfortably below the uint64 prime ceiling.
            boot_prime_bits = min(logscale + 15, 61)
            boot_primes = _gen_primes(
                [boot_prime_bits] * num_reserved, logn, used)
            main_primes = main_primes + boot_primes

        aux_primes = _gen_primes(logp, logn, used)

        max_level = len(main_primes) - 1
        print(f"[Cheddar] Creating CKKS context: LogN={logn}, "
              f"LogQ={list(logq)}, LogP={list(logp)}, scale=2^{logscale}, "
              f"slots={self._slots}, word=uint64, "
              f"hoist_mode={self._hoist_mode_name}, "
              f"boot={'on' if self._boot_enabled else 'off'}, "
              f"max_level={max_level}, default_enc_level={default_enc_level}")

        setup_args = {
            "LogN": logn,
            "LogScale": logscale,
            "MainPrimes": main_primes,
            "AuxPrimes": aux_primes,
            "DefaultEncryptionLevel": default_enc_level,
        }
        if self._boot_enabled:
            # Presence of these keys tells the native layer to build the
            # scheme's single context as a BootContext (see setup_scheme in
            # cheddar_pybind.cu). Using one context rather than a separate
            # regular Context + BootContext roughly halves resident memory
            # -- the difference between fitting and OOMing full-slot
            # LogN=16 bootstrap on a 24GB card. Must satisfy default_enc_level
            # == max_level - num_cts_levels - eval_mod_levels, which the
            # chain extension above guarantees by construction.
            setup_args["BootNumCtsLevels"] = self._boot_num_cts_levels
            setup_args["BootNumStcLevels"] = self._boot_num_stc_levels
            setup_args["BootLogMessageRatio"] = self._boot_log_message_ratio
            # min_ks: full boot key set by default (fastest). Set
            # ORION_CHEDDAR_BOOT_MIN_KS=1 to trade Boot speed for far less
            # rotation-key memory -- needed when many boot circuits (several
            # slot counts) plus linear-transform keys must be resident at
            # once, e.g. ResNet on a 24GB card.
            env = os.environ.get("ORION_CHEDDAR_BOOT_MIN_KS", "").lower()
            setup_args["BootMinKs"] = env in ("1", "true", "yes", "on")

        self._generation = _native.setup_scheme(setup_args)

    def DeleteScheme(self) -> None:
        if self._generation is not None:
            _native.delete_scheme_if(self._generation)
            self._generation = None

    # ------------------------------------------------------------------
    # setup_*  (sub-steps called by setup_bindings; the native module
    # holds a single UserInterface that covers keygen/encrypt/decrypt)
    # ------------------------------------------------------------------

    def setup_tensor_binds(self) -> None:
        """No-op: Cheddar objects live in std::map registries in C++."""

    def setup_key_generator(self) -> None:
        """No-op. Orion's key_generator calls Generate*Key below."""

    def setup_encoder(self) -> None:
        """No-op. Encoder lives inside the Cheddar Context."""

    def setup_encryptor(self) -> None:
        """No-op. UserInterface covers encrypt/decrypt."""

    def setup_evaluator(self) -> None:
        """No-op. Ops are free functions on the native Context."""

    # setup_poly_evaluator: defined in the Polynomial evaluation section
    # below.
    # setup_lt_evaluator: defined in the Linear transform section below.

    def setup_bootstrapper(self) -> None:
        """No-op. NewBootstrapper lazily builds Cheddar's BootContext on
        first use -- most configs (e.g. lola/mlp) never call it."""

    # ------------------------------------------------------------------
    # Key generation. UserInterface's constructor samples secrets and
    # builds the basic evks, so everything happens in GenerateSecretKey.
    # ------------------------------------------------------------------

    def NewKeyGenerator(self) -> None:
        _native.NewKeyGenerator()

    def GenerateSecretKey(self) -> None:
        _native.GenerateSecretKey()

    def GeneratePublicKey(self) -> None:
        _native.GeneratePublicKey()

    def GenerateRelinearizationKey(self) -> None:
        _native.GenerateRelinearizationKey()

    def GenerateEvaluationKeys(self) -> None:
        # Rotation keys are generated lazily per shift via AddRotationKey.
        _native.GenerateEvaluationKeys()

    # ------------------------------------------------------------------
    # Encoder / Encryptor factory methods
    # ------------------------------------------------------------------

    def NewEncoder(self) -> None:
        """No-op. Encoder is part of the Cheddar Context."""

    def NewEncryptor(self) -> None:
        _native.NewEncryptor()

    def NewDecryptor(self) -> None:
        _native.NewDecryptor()

    def NewEvaluator(self) -> None:
        """No-op."""

    # NewPolynomialEvaluator: defined in the Polynomial evaluation section
    # below.

    # NewLinearTransformEvaluator: defined in the Linear transform section
    # below.

    def DeleteBootstrappers(self) -> None:
        _native.DeleteBootstrappers()

    # ------------------------------------------------------------------
    # Encode / Encrypt / Decode / Decrypt
    # ------------------------------------------------------------------

    def Encode(self, values, level: int, scale) -> int:
        return _native.Encode(list(values), int(level), float(scale))

    def Decode(self, pt_id: int):
        return _native.Decode(int(pt_id))

    def Encrypt(self, pt_id: int) -> int:
        return _native.Encrypt(int(pt_id))

    def Decrypt(self, ct_id: int) -> int:
        return _native.Decrypt(int(ct_id))

    # ------------------------------------------------------------------
    # ct - ct arithmetic
    # ------------------------------------------------------------------

    def AddCiphertextNew(self, a: int, b: int) -> int:
        return _native.AddCiphertextNew(int(a), int(b))

    def AddCiphertext(self, a: int, b: int) -> int:
        return _native.AddCiphertext(int(a), int(b))

    def SubCiphertextNew(self, a: int, b: int) -> int:
        return _native.SubCiphertextNew(int(a), int(b))

    def SubCiphertext(self, a: int, b: int) -> int:
        return _native.SubCiphertext(int(a), int(b))

    def MulRelinCiphertextNew(self, a: int, b: int) -> int:
        return _native.MulRelinCiphertextNew(int(a), int(b))

    def MulRelinCiphertext(self, a: int, b: int) -> int:
        return _native.MulRelinCiphertext(int(a), int(b))

    def MulNoRelinCiphertextNew(self, a: int, b: int) -> int:
        return _native.MulNoRelinCiphertextNew(int(a), int(b))

    def RelinearizeNew(self, a: int) -> int:
        return _native.RelinearizeNew(int(a))

    def Negate(self, ct_id: int) -> int:
        return _native.Negate(int(ct_id))

    # ------------------------------------------------------------------
    # ct - pt arithmetic
    # ------------------------------------------------------------------

    def MulPlaintextNew(self, ct_id: int, pt_id: int) -> int:
        return _native.MulPlaintextNew(int(ct_id), int(pt_id))

    def MulPlaintext(self, ct_id: int, pt_id: int) -> int:
        return _native.MulPlaintext(int(ct_id), int(pt_id))

    def AddPlaintextNew(self, ct_id: int, pt_id: int) -> int:
        return _native.AddPlaintextNew(int(ct_id), int(pt_id))

    def AddPlaintext(self, ct_id: int, pt_id: int) -> int:
        return _native.AddPlaintext(int(ct_id), int(pt_id))

    def SubPlaintextNew(self, ct_id: int, pt_id: int) -> int:
        return _native.SubPlaintextNew(int(ct_id), int(pt_id))

    def SubPlaintext(self, ct_id: int, pt_id: int) -> int:
        return _native.SubPlaintext(int(ct_id), int(pt_id))

    # ------------------------------------------------------------------
    # ct - scalar arithmetic
    # ------------------------------------------------------------------

    def AddScalarNew(self, ct_id: int, scalar) -> int:
        return _native.AddScalarNew(int(ct_id), float(scalar))

    def AddScalar(self, ct_id: int, scalar) -> int:
        return _native.AddScalar(int(ct_id), float(scalar))

    def SubScalarNew(self, ct_id: int, scalar) -> int:
        return _native.SubScalarNew(int(ct_id), float(scalar))

    def SubScalar(self, ct_id: int, scalar) -> int:
        return _native.SubScalar(int(ct_id), float(scalar))

    def MulScalarFloatNew(self, ct_id: int, scalar) -> int:
        return _native.MulScalarFloatNew(int(ct_id), float(scalar))

    def MulScalarFloat(self, ct_id: int, scalar) -> int:
        return _native.MulScalarFloat(int(ct_id), float(scalar))

    def MulScalarIntNew(self, ct_id: int, scalar) -> int:
        return _native.MulScalarIntNew(int(ct_id), int(scalar))

    def MulScalarInt(self, ct_id: int, scalar) -> int:
        return _native.MulScalarInt(int(ct_id), int(scalar))

    # ------------------------------------------------------------------
    # Rescale
    # ------------------------------------------------------------------

    def RescaleNew(self, ct_id: int) -> int:
        return _native.RescaleNew(int(ct_id))

    def Rescale(self, ct_id: int) -> int:
        return _native.Rescale(int(ct_id))

    # ------------------------------------------------------------------
    # Rotations
    # ------------------------------------------------------------------

    def AddRotationKey(self, k: int) -> None:
        _native.AddRotationKey(int(k))

    def RotateNew(self, ct_id: int, k: int) -> int:
        # Keys are generated lazily inside the native RotateNew.
        return _native.RotateNew(int(ct_id), int(k))

    def RotateBatchNew(self, ct_id: int, ks: Sequence[int]) -> list[int]:
        """N rotations of one ciphertext.

        hoist_mode "none" runs a per-shift HRot loop; "single" shares
        one ModUp across the batch.
        """
        ks = [int(k) for k in ks]
        return _native.RotateBatchNew(int(ct_id), ks, self._hoist_mode)

    # ------------------------------------------------------------------
    # Lifecycle: ID deletion
    # ------------------------------------------------------------------

    def DeleteCiphertext(self, ct_id: int) -> None:
        _native.DeleteCiphertext(int(ct_id))

    def DeletePlaintext(self, pt_id: int) -> None:
        _native.DeletePlaintext(int(pt_id))

    # ------------------------------------------------------------------
    # Linear transform (diagonal matrix-vector multiply)
    #
    # No native LT object -- Cheddar's own HoistHandler double-hoisting
    # is built around a PlainHoistMap constructor we haven't wired up
    # (see docs on HoistHandler in the vendored source). This evaluates
    # each diagonal via rotate + ct-pt multiply + add instead, same
    # approach as DeSiLoLibrary: naive O(N) loop, or BSGS O(2*sqrt(N))
    # when bsgs_ratio requests it and there's more than one diagonal.
    #
    # Rotation convention: RotateNew(ct, k)[i] = ct[i+k] (see the
    # "Convention check" note in cheddar_pybind.cu). The diag-k
    # semantics this backend is tested against are
    # result[i] = sum_k diag_k[i] * input[(i+k) % slots], so baby/giant
    # rotations use +distance directly -- no sign flip (DeSiLoLibrary's
    # naive/BSGS loop negates every distance because its native rotate
    # uses the opposite convention).
    # ------------------------------------------------------------------

    def setup_lt_evaluator(self) -> None:
        """No-op. LT state (self._transforms) is already set up in __init__."""

    def NewLinearTransformEvaluator(self) -> None:
        """No-op."""

    def GenerateLinearTransform(self, diags_idxs, diags_data, level,
                                bsgs_ratio, io_mode) -> int:
        num_diags = len(diags_idxs)
        values_per_diag = len(diags_data) // num_diags
        diags: dict[int, list[float]] = {}
        for i, idx in enumerate(diags_idxs):
            start = i * values_per_diag
            vals = list(diags_data[start:start + values_per_diag])
            if len(vals) < self._slots:
                vals = vals + [0.0] * (self._slots - len(vals))
            diags[int(idx)] = vals[:self._slots]

        lt_id = self._next_lt_id
        self._next_lt_id += 1
        self._transforms[lt_id] = {"diags": diags, "bsgs_ratio": bsgs_ratio}
        return lt_id

    def _lt_use_bsgs(self, lt_id: int) -> bool:
        t = self._transforms[lt_id]
        return t["bsgs_ratio"] not in ("none", None) and len(t["diags"]) > 1

    def _lt_bsgs_stride(self, lt_id: int) -> int:
        t = self._transforms[lt_id]
        return max(1, math.ceil(
            math.sqrt(len(t["diags"]) * float(t["bsgs_ratio"]))))

    def GetLinearTransformRotationKeys(self, lt_id: int) -> list[int]:
        diags = self._transforms[lt_id]["diags"]
        if not self._lt_use_bsgs(lt_id):
            return sorted(k for k in diags if k != 0)
        bs = self._lt_bsgs_stride(lt_id)
        dists = set()
        for k in diags:
            b, g = k % bs, k // bs
            if b != 0:
                dists.add(b)
            if g != 0:
                dists.add(g * bs)
        return sorted(dists)

    def GenerateLinearTransformRotationKey(self, k: int) -> None:
        self.AddRotationKey(int(k))

    def EvaluateLinearTransform(self, lt_id: int, ct_id: int) -> int:
        diags = self._transforms[lt_id]["diags"]
        if self._lt_use_bsgs(lt_id):
            return self._eval_lt_bsgs(ct_id, diags, self._lt_bsgs_stride(lt_id))
        return self._eval_lt_naive(ct_id, diags)

    def _lt_encode(self, values, level: int) -> int:
        return self.Encode(values, level, self._default_scale)

    def _eval_lt_naive(self, ct_id: int, diags: dict[int, list[float]]) -> int:
        level = self.GetCiphertextLevel(ct_id)
        result = None
        for k, vals in diags.items():
            rotated = ct_id if k == 0 else self.RotateNew(ct_id, k)
            pt = self._lt_encode(vals, level)
            prod = self.MulPlaintextNew(rotated, pt)
            self.DeletePlaintext(pt)
            if rotated != ct_id:
                self.DeleteCiphertext(rotated)
            result = prod if result is None else self._lt_accumulate(result, prod)
        return result

    def _eval_lt_bsgs(self, ct_id: int, diags: dict[int, list[float]],
                      bs: int) -> int:
        level = self.GetCiphertextLevel(ct_id)

        giant_groups: dict[int, list[tuple[int, list[float]]]] = {}
        for k, vals in diags.items():
            b, g = k % bs, k // bs
            giant_groups.setdefault(g, []).append((b, vals))

        needed_babies = sorted({b for group in giant_groups.values()
                                for b, _ in group})
        baby_rots = {b: (ct_id if b == 0 else self.RotateNew(ct_id, b))
                    for b in needed_babies}

        result = None
        for g, group in giant_groups.items():
            inner = None
            for b, vals in group:
                shifted = vals if g == 0 else np.roll(vals, g * bs).tolist()
                pt = self._lt_encode(shifted, level)
                prod = self.MulPlaintextNew(baby_rots[b], pt)
                self.DeletePlaintext(pt)
                inner = prod if inner is None else self._lt_accumulate(inner, prod)

            partial = inner if g == 0 else self.RotateNew(inner, g * bs)
            if partial is not inner:
                self.DeleteCiphertext(inner)
            result = partial if result is None else self._lt_accumulate(result, partial)

        for b, rot in baby_rots.items():
            if rot != ct_id:
                self.DeleteCiphertext(rot)
        return result

    def _lt_accumulate(self, a: int, b: int) -> int:
        out = self.AddCiphertextNew(a, b)
        self.DeleteCiphertext(a)
        self.DeleteCiphertext(b)
        return out

    def DeleteLinearTransform(self, lt_id: int) -> None:
        self._transforms.pop(lt_id, None)

    def RemoveRotationKeys(self) -> None:
        """No-op. Cheddar caches rotation keys natively; no release path."""

    def RemovePlaintextDiagonals(self, lt_id: int) -> None:
        """No-op. No native diagonal storage to release."""

    def GenerateAndSerializeRotationKey(self, k):
        self._not_implemented("GenerateAndSerializeRotationKey")

    def LoadRotationKey(self, byte_data, k=None):
        self._not_implemented("LoadRotationKey")

    def SerializeDiagonal(self, lt_id, diag_idx):
        self._not_implemented("SerializeDiagonal")

    def LoadPlaintextDiagonal(self, byte_data, lt_id, diag_idx):
        self._not_implemented("LoadPlaintextDiagonal")

    # ------------------------------------------------------------------
    # Polynomial evaluation
    #
    # No native evaluator -- GenerateMonomial/GenerateChebyshev just store
    # coefficients; EvaluatePolynomial runs Horner's method over ct-ct
    # multiply (MulRelinCiphertext + Rescale) and ct-scalar ops. Chebyshev
    # coefficients are converted to monomial basis via numpy's cheb2poly
    # before evaluation (same fallback DeSiLoLibrary uses) rather than a
    # ciphertext-level Clenshaw recurrence -- fine at the low degrees this
    # is tested at; a dedicated Chebyshev evaluator would be needed for
    # high-degree/high-precision approximations (e.g. sign/ReLU).
    # ------------------------------------------------------------------

    def setup_poly_evaluator(self) -> None:
        """No-op. Polynomial state (self._polynomials) is set up in __init__."""

    def NewPolynomialEvaluator(self) -> None:
        """No-op."""

    def GenerateMonomial(self, coeffs) -> int:
        """coeffs: ascending [a0, a1, ..., an] (evaluator.py already
        reversed the caller's descending order before calling this)."""
        poly_id = self._next_poly_id
        self._next_poly_id += 1
        self._polynomials[poly_id] = (list(coeffs), "monomial")
        return poly_id

    def GenerateChebyshev(self, coeffs) -> int:
        """coeffs: [c0, c1, ..., cn] for f(x) = sum_j c_j * T_j(x)."""
        poly_id = self._next_poly_id
        self._next_poly_id += 1
        self._polynomials[poly_id] = (list(coeffs), "chebyshev")
        return poly_id

    def EvaluatePolynomial(self, ct_id: int, poly_id: int, scale) -> int:
        coeffs, kind = self._polynomials[poly_id]
        if kind == "chebyshev":
            coeffs = np.polynomial.chebyshev.cheb2poly(coeffs).tolist()
        return self._eval_monomial(ct_id, coeffs)

    def _eval_monomial(self, ct_id: int, coeffs: list[float]) -> int:
        """Horner's method: f(x) = a0 + x*(a1 + x*(... + x*an)).
        coeffs ascending [a0, ..., an]."""
        n = len(coeffs) - 1
        if n < 0:
            raise ValueError("EvaluatePolynomial: empty coefficient list.")
        if n == 0:
            acc = self.Rescale(self.MulScalarFloatNew(ct_id, 0.0))
            return self.AddScalar(acc, coeffs[0])

        acc = self.Rescale(self.MulScalarFloatNew(ct_id, coeffs[n]))
        acc = self.AddScalar(acc, coeffs[n - 1])
        for i in range(n - 2, -1, -1):
            prod = self.MulRelinCiphertextNew(acc, ct_id)
            self.DeleteCiphertext(acc)
            acc = self.Rescale(prod)
            acc = self.AddScalar(acc, coeffs[i])
        return acc

    @staticmethod
    def _fit_chebyshev_standard_basis(x, y, degree):
        """Least-squares fit in the standard Chebyshev basis T_n(x). No
        domain mapping (unlike numpy's Chebyshev.fit): the returned
        coefficients c satisfy sum_j c[j] * T_j(x) ~= y for the actual x
        values given."""
        n = degree + 1
        T = np.zeros((len(x), n))
        T[:, 0] = 1.0
        if n > 1:
            T[:, 1] = x
        for j in range(2, n):
            T[:, j] = 2.0 * x * T[:, j - 1] - T[:, j - 2]
        coeffs, _, _, _ = np.linalg.lstsq(T, y, rcond=None)
        return coeffs

    @staticmethod
    def _eval_chebyshev_standard_basis(x, coeffs):
        n = len(coeffs)
        T = np.zeros((len(x), n))
        T[:, 0] = 1.0
        if n > 1:
            T[:, 1] = x
        for j in range(2, n):
            T[:, j] = 2.0 * x * T[:, j - 1] - T[:, j - 2]
        return T @ coeffs

    def GenerateMinimaxSignCoeffs(self, degrees, prec, logalpha, logerr,
                                  debug) -> list[float]:
        """Composite sign-polynomial coefficients, pure numpy -- no FHE
        ops, so this is identical in spirit to DeSiLoLibrary's version.
        Each stage is fitted on the *output range* of the previous stage
        (matching Lattigo's composite Remez strategy) so the composition
        converges toward sign(x). The last polynomial approximates
        step(x) in {0, 1} (Lattigo convention: ReLU = x * step(x)); every
        earlier one approximates sign(x) in {-1, 1}.
        """
        gap = 2.0 ** (-logalpha)
        domain_min, domain_max = -1.0, 1.0
        current_gap = gap

        coeffs_flat: list[float] = []
        for i, deg in enumerate(degrees):
            is_last = (i == len(degrees) - 1)
            n_pts = max(8 * deg, 500)

            x_neg = np.linspace(domain_min, -current_gap, n_pts)
            x_pos = np.linspace(current_gap, domain_max, n_pts)
            x = np.concatenate([x_neg, x_pos])
            y = np.where(x > 0, 1.0, 0.0) if is_last else np.sign(x)

            coeffs = self._fit_chebyshev_standard_basis(x, y, deg)
            coeffs_flat.extend(coeffs[:deg + 1].tolist())

            if not is_last:
                dense_neg = np.linspace(domain_min, -current_gap, 2000)
                dense_pos = np.linspace(current_gap, domain_max, 2000)
                dense = np.concatenate([dense_neg, dense_pos])
                output = self._eval_chebyshev_standard_basis(dense, coeffs)

                domain_min = float(output.min())
                domain_max = float(output.max())

                near_gap = np.array([current_gap, -current_gap])
                near_out = self._eval_chebyshev_standard_basis(near_gap, coeffs)
                current_gap = max(
                    float(min(abs(near_out[0]), abs(near_out[1]))), 1e-12)

        return coeffs_flat

    # ------------------------------------------------------------------
    # Metadata getters
    # ------------------------------------------------------------------

    def GetKeyMemoryMB(self) -> float:
        """Exact resident evaluation-key memory (rotation + basic evks)."""
        return _native.GetKeyMemoryMB()

    def GetModuliChain(self) -> list[int]:
        """Usable-level moduli (q_i), indexed by Orion level 0..max_level.

        Callers (bootstrap prescale, batch norm, extract/embedding) encode
        plaintexts at scale = q[level] so the following rescale is
        errorless. Excludes the reserved boot-circuit primes -- those sit
        above Orion's level range. Returns the actual primes (not a
        default-scale stub as desilo does) since cheddar rescales by the
        exact prime, so the real value is what keeps rescaling clean."""
        return list(self._usable_primes)

    def GetAuxModuliChain(self) -> list[int]:
        return []

    def GetCiphertextLevel(self, ct_id: int) -> int:
        return _native.GetCiphertextLevel(int(ct_id))

    def GetPlaintextLevel(self, pt_id: int) -> int:
        return _native.GetPlaintextLevel(int(pt_id))

    def GetCiphertextSlots(self, ct_id: int) -> int:
        return _native.GetCiphertextSlots(int(ct_id))

    def GetPlaintextSlots(self, pt_id: int) -> int:
        return _native.GetPlaintextSlots(int(pt_id))

    def GetCiphertextScale(self, ct_id: int) -> float:
        return _native.GetCiphertextScale(int(ct_id))

    def GetPlaintextScale(self, pt_id: int) -> float:
        return _native.GetPlaintextScale(int(pt_id))

    def SetCiphertextScale(self, ct_id: int, scale) -> None:
        _native.SetCiphertextScale(int(ct_id), float(scale))

    def SetPlaintextScale(self, pt_id: int, scale) -> None:
        _native.SetPlaintextScale(int(pt_id), float(scale))

    # ------------------------------------------------------------------
    # Not yet implemented
    # ------------------------------------------------------------------

    def _not_implemented(self, name: str):
        raise NotImplementedError(
            f"CheddarLibrary.{name}() is not implemented yet. Use the "
            f"desilo or lattigo backend, or extend the wrap in "
            f"orion/backend/cheddar/."
        )

    # ------------------------------------------------------------------
    #  Bootstrap
    # ------------------------------------------------------------------

    def NewBootstrapper(self, logPs, slots: int) -> None:
        """Prepare Cheddar's native BootContext for the given slot count.

        logPs (lattigo's auxiliary-prime sizing knob) is ignored -- see
        the comment in setup_scheme. Idempotent per slot count: the
        BootContext and its EvalMod are built once and reused; only
        PrepareEvalSpecialFFT + rotation-key generation repeat for a new
        slot count.
        """
        print(f"[Cheddar] Preparing bootstrapper for slots={slots} ...")
        _native.NewBootstrapper(
            self._boot_num_cts_levels, self._boot_num_stc_levels,
            self._boot_log_message_ratio, int(slots))
        print(f"[Cheddar] Bootstrapper ready.")

    def Bootstrap(self, ct_id: int, slots: int) -> int:
        return _native.Bootstrap(int(ct_id), int(slots))
