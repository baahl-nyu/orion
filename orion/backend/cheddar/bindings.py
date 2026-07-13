"""Cheddar CKKS backend.

Mirrors the same ID-based interface as DeSiLoLibrary / LattigoLibrary so
that orion/backend/python/ keeps working unchanged. All actual FHE work
goes through the native ``_cheddar_native`` pybind11 module built from
``orion/backend/cheddar/ext/``.

Not yet implemented: polynomial evaluator, BSGS linear transform, and
bootstrap -- attempting to use them raises NotImplementedError. Extending
these is the path to running this backend against the full test suite
and the run_lola/run_mlp/run_resnet examples.

Word size: uint64. Cheddar's Parameter takes explicit prime lists rather
than bit sizes, so this module converts Orion's LogQ/LogP into
NTT-friendly primes (p = 1 mod 2N) before calling setup_scheme.
"""

from __future__ import annotations

import atexit
from typing import Sequence

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
        # Generation this instance owns; DeleteScheme only tears down
        # native state if it is still the live generation (the bench
        # runner GC-finalizes the OLD scheme after the NEW one is set
        # up -- an unconditional delete would wipe the new state).
        self._generation: int | None = None

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

        if orion_params.get_boot_logp() != logp:
            raise NotImplementedError(
                "Cheddar backend doesn't wrap bootstrapping. Use desilo "
                "or lattigo for bootstrap-dependent configs."
            )

        used: set[int] = set()
        main_primes = _gen_primes(logq, logn, used)
        aux_primes = _gen_primes(logp, logn, used)

        print(f"[Cheddar] Creating CKKS context: LogN={logn}, "
              f"LogQ={list(logq)}, LogP={list(logp)}, scale=2^{logscale}, "
              f"slots={self._slots}, word=uint64, "
              f"hoist_mode={self._hoist_mode_name}")

        self._generation = _native.setup_scheme({
            "LogN": logn,
            "LogScale": logscale,
            "MainPrimes": main_primes,
            "AuxPrimes": aux_primes,
        })

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

    def setup_poly_evaluator(self) -> None:
        """No-op. Polynomial eval not yet implemented."""

    def setup_lt_evaluator(self) -> None:
        """No-op. Linear-transform eval not yet implemented."""

    def setup_bootstrapper(self) -> None:
        """No-op. Bootstrap not yet implemented."""

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

    def NewPolynomialEvaluator(self) -> None:
        """No-op. Polynomial eval not yet implemented."""

    def NewLinearTransformEvaluator(self) -> None:
        """No-op. Linear-transform eval not yet implemented."""

    def DeleteBootstrappers(self) -> None:
        """No-op. Bootstrap not yet implemented; nothing allocated to free."""

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

    def SubCiphertextNew(self, a: int, b: int) -> int:
        return _native.SubCiphertextNew(int(a), int(b))

    def MulRelinCiphertextNew(self, a: int, b: int) -> int:
        return _native.MulRelinCiphertextNew(int(a), int(b))

    def MulNoRelinCiphertextNew(self, a: int, b: int) -> int:
        return _native.MulNoRelinCiphertextNew(int(a), int(b))

    def RelinearizeNew(self, a: int) -> int:
        return _native.RelinearizeNew(int(a))

    # ------------------------------------------------------------------
    # ct - pt arithmetic
    # ------------------------------------------------------------------

    def MulPlaintextNew(self, ct_id: int, pt_id: int) -> int:
        return _native.MulPlaintextNew(int(ct_id), int(pt_id))

    def AddPlaintextNew(self, ct_id: int, pt_id: int) -> int:
        return _native.AddPlaintextNew(int(ct_id), int(pt_id))

    def SubPlaintextNew(self, ct_id: int, pt_id: int) -> int:
        return _native.SubPlaintextNew(int(ct_id), int(pt_id))

    # ------------------------------------------------------------------
    # Rescale
    # ------------------------------------------------------------------

    def RescaleNew(self, ct_id: int) -> int:
        return _native.RescaleNew(int(ct_id))

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
    # Metadata getters
    # ------------------------------------------------------------------

    def GetKeyMemoryMB(self) -> float:
        """Exact resident evaluation-key memory (rotation + basic evks)."""
        return _native.GetKeyMemoryMB()

    def GetCiphertextLevel(self, ct_id: int) -> int:
        return _native.GetCiphertextLevel(int(ct_id))

    def GetPlaintextLevel(self, pt_id: int) -> int:
        return _native.GetPlaintextLevel(int(pt_id))

    def GetCiphertextSlots(self, ct_id: int) -> int:
        return _native.GetCiphertextSlots(int(ct_id))

    def GetPlaintextSlots(self, pt_id: int) -> int:
        return _native.GetPlaintextSlots(int(pt_id))

    # ------------------------------------------------------------------
    # Not yet implemented
    # ------------------------------------------------------------------

    def _not_implemented(self, name: str):
        raise NotImplementedError(
            f"CheddarLibrary.{name}() is not implemented yet. Use the "
            f"desilo or lattigo backend, or extend the wrap in "
            f"orion/backend/cheddar/."
        )

    def NewBootstrapper(self, *args, **kwargs):
        self._not_implemented("NewBootstrapper")

    def Bootstrap(self, *args, **kwargs):
        self._not_implemented("Bootstrap")
