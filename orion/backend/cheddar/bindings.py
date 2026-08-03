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

# ORION_CHEDDAR_WORD32=1: use the uint32_t twin module. ~30-bit primes,
# rational rescaling -- Cheddar's native design point (its paper's 0.72s
# ResNet numbers are 32-bit), halving the memory footprint of every
# ciphertext/key/plaintext. Requires a pregenerated chain file (see
# params32/ and ORION_CHEDDAR_PARAMS_JSON below); LogQ/LogP bit lists
# cannot express these chains.
_WORD32 = os.environ.get("ORION_CHEDDAR_WORD32", "").lower() in (
    "1", "true", "yes", "on")

try:
    if _WORD32:
        from . import _cheddar_native32 as _native
    else:
        from . import _cheddar_native as _native
except ImportError as e:  # pragma: no cover - import-time diagnostic
    raise ImportError(
        f"Could not import {'_cheddar_native32' if _WORD32 else '_cheddar_native'}. "
        "Build the extension via the CMake steps in "
        "orion/backend/cheddar/README.md. "
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

        # ORION_CHEDDAR_SHARED_BS=<int>: use ONE fixed BSGS baby-step stride
        # for every linear-transform block network-wide, instead of each
        # layer independently picking bs = ceil(sqrt(num_diags*bsgs_ratio))
        # (locally optimal per layer, but produces incommensurate giant-step
        # distances -- multiples of a different bs per layer -- so distinct
        # rotation keys almost never overlap across layers). A shared bs
        # makes baby keys (0..bs-1, already cheap/shared) and giant keys
        # (multiples of the SAME bs everywhere) both eligible for cross-layer
        # reuse via the existing global key cache. Mathematically safe for
        # any bs >= 1 (b, g = k % bs, k // bs always partitions correctly);
        # this is a pure memory/reuse tradeoff, not a correctness knob.
        # Measured on ResNet20 (exact-key mode): bs=64 cuts total distinct
        # rotation keys network-wide from 362 to 134 (~2.7x). Default unset:
        # unchanged per-layer-independent behavior.
        shared_bs_env = os.environ.get("ORION_CHEDDAR_SHARED_BS", "").strip()
        self._shared_bs = int(shared_bs_env) if shared_bs_env else None

        # ORION_CHEDDAR_DOUBLE_HOIST=1: evaluate every linear-transform
        # block through the native HoistHandler (Bossuat et al.
        # EUROCRYPT'21 double hoisting: fused baby-step
        # KeyMult+MAC+automorphism sharing one ModUp, fused giant-step
        # accumulation, final ModDown with the rescale folded in) instead
        # of the Python-orchestrated BSGS loop. The native output is
        # already rescaled -- lt_output_prerescaled tells lt_evaluator to
        # skip its usual post-LT rescale. Off by default until verified.
        dh_env = os.environ.get("ORION_CHEDDAR_DOUBLE_HOIST", "").lower()
        self._double_hoist = dh_env in ("1", "true", "yes", "on")
        self.lt_output_prerescaled = self._double_hoist

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
        # Only num_cts_levels + eval_mod_levels primes are reserved above
        # the usable chain -- StC eats into the usable chain itself, so
        # Boot() restores to at most l_eff - num_stc_levels rather than
        # l_eff. Deliberate: giving StC its own reserved primes too (so
        # Boot() restores fully to l_eff) works and cuts ResNet20's
        # bootstrap count nearly in half (75 -> 38, see BootContext.cpp's
        # comment on the GetStCStartLevel()/GetEndLevel() anchor), but it's
        # a net latency loss -- measured 51.4s (38 boots, full restore) vs
        # 36.0s (75 boots, this reservation) on ResNet20/cheddar/LogN=16.
        # More, cheaper bootstraps beat fewer, pricier ones plus wider
        # downstream ciphertexts here.
        raw_cts = orion_params.get_boot_num_cts_levels()
        raw_stc = orion_params.get_boot_num_stc_levels()
        self._boot_enabled = raw_cts is not None and raw_stc is not None
        self._boot_num_cts_levels = raw_cts or 4
        self._boot_num_stc_levels = raw_stc or 3
        self._boot_log_message_ratio = (
            orion_params.get_boot_log_message_ratio() or 5)

        if _WORD32:
            self._setup_scheme_word32(logq, logp, logscale, logn)
            return

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
            # == max_level - num_cts_levels - eval_mod_levels - num_stc_levels
            # (== BootContext's GetEndLevel()), which the chain extension
            # above guarantees by construction.
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

    def _setup_scheme_word32(self, logq, logp, logscale, logn) -> None:
        """32-bit-word setup from a pregenerated chain file.

        ~2^40 scales cannot be one ~30-bit prime per level, so word32
        chains use rational rescaling: each level's rescale divides by a
        multi-prime product (main + rotating terminal primes) that only
        approximates 2^LogScale, described by an explicit LevelConfig.
        These are pregenerated (Cheddar ships reference chains in
        parameters/; ours is vendored under params32/) and passed through
        verbatim -- the YAML's LogQ/LogP entries are nominal placeholders
        whose only meaningful property is their COUNT, which must match
        the chain so Orion's level accounting (bootstrap placement, depth
        budgeting) agrees with the native scheme.
        """
        import json

        json_path = os.environ.get("ORION_CHEDDAR_PARAMS_JSON", "").strip()
        if not json_path:
            json_path = os.path.join(
                os.path.dirname(__file__), "params32", "bootparam_40.json")
        with open(json_path) as f:
            chain = json.load(f)

        if logn != chain["log_degree"]:
            raise ValueError(
                f"word32: YAML LogN={logn} != chain log_degree="
                f"{chain['log_degree']} ({json_path})")
        if logscale != chain["log_default_scale"]:
            raise ValueError(
                f"word32: YAML LogScale={logscale} != chain "
                f"log_default_scale={chain['log_default_scale']} ({json_path})")
        enc_level = int(chain["default_encryption_level"])
        if len(logq) - 1 != enc_level:
            raise ValueError(
                f"word32: YAML LogQ has {len(logq)} entries (usable max "
                f"level {len(logq) - 1}) but the chain's "
                f"default_encryption_level is {enc_level}. The chain is "
                f"rigid (boot circuit levels are baked in), so the YAML "
                f"must declare exactly {enc_level + 1} LogQ entries "
                f"({json_path}).")

        chain_boot = bool(chain.get("boot", False))
        if self._boot_enabled and not chain_boot:
            raise ValueError(
                f"word32: YAML requests bootstrap but chain {json_path} "
                f"has boot=false.")
        # Filled in after native setup (needs the live Parameter): see
        # the GetCanonicalScales call below.
        self._usable_primes = []

        max_level = len(chain["level_config"]) - 1
        print(f"[Cheddar] Creating CKKS context (word32): LogN={logn}, "
              f"chain={os.path.basename(json_path)}, scale=2^{logscale}, "
              f"slots={self._slots}, hoist_mode={self._hoist_mode_name}, "
              f"boot={'on' if chain_boot else 'off'}, "
              f"max_level={max_level}, default_enc_level={enc_level}")

        setup_args = {
            "LogN": logn,
            "LogScale": logscale,
            "MainPrimes": list(chain["main_primes"]),
            "AuxPrimes": list(chain["auxiliary_primes"]),
            "TerPrimes": list(chain.get("terminal_primes", [])),
            "LevelConfig": [list(p) for p in chain["level_config"]],
            "AdditionalBase": list(chain.get("additional_base", [0, 0])),
            "DefaultEncryptionLevel": enc_level,
        }
        if "dense_hamming_weight" in chain:
            setup_args["DenseHammingWeight"] = int(
                chain["dense_hamming_weight"])
        if "sparse_hamming_weight" in chain:
            setup_args["SparseHammingWeight"] = int(
                chain["sparse_hamming_weight"])

        if chain_boot:
            # Boot geometry comes from the chain (its reserved levels are
            # part of the pregenerated structure); YAML knobs must agree
            # if present.
            cts = int(chain["num_cts_levels"])
            stc = int(chain["num_stc_levels"])
            if self._boot_enabled and (
                    self._boot_num_cts_levels != cts
                    or self._boot_num_stc_levels != stc):
                raise ValueError(
                    f"word32: YAML boot levels (cts="
                    f"{self._boot_num_cts_levels}, stc="
                    f"{self._boot_num_stc_levels}) != chain (cts={cts}, "
                    f"stc={stc}) ({json_path}).")
            self._boot_enabled = True
            self._boot_num_cts_levels = cts
            self._boot_num_stc_levels = stc
            setup_args["BootNumCtsLevels"] = cts
            setup_args["BootNumStcLevels"] = stc
            setup_args["BootLogMessageRatio"] = self._boot_log_message_ratio
            env = os.environ.get("ORION_CHEDDAR_BOOT_MIN_KS", "").lower()
            setup_args["BootMinKs"] = env in ("1", "true", "yes", "on")

        self._generation = _native.setup_scheme(setup_args)

        # GetModuliChain feeds Orion's errorless-rescale idiom (nn layers
        # encode plaintexts at moduli_chain[level] so mult-then-rescale
        # stays scale-exact). With rational rescaling the value that
        # achieves this is the level's CANONICAL scale, not any prime:
        # canonical(l)^2 / rescale_prod(l) == canonical(l-1), so a
        # plaintext at canonical(l) keeps the ciphertext on the
        # trajectory through both ct-pt and ct-ct multiplies. Rounded to
        # int here (GetModuliChain's contract); the native Encode snaps
        # near-canonical requests onto the exact double.
        self._usable_primes = [
            int(round(s)) for s in _native.GetCanonicalScales()]

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

    def MulPlaintextAccumNew(self, ct_ids: Sequence[int],
                             pt_ids: Sequence[int]) -> int:
        """Fused MAC: sum_i ct_i * pt_i in one native PAccum kernel chain
        instead of N ct-pt mults + N-1 adds (each a separate memory-bound
        pass over full ciphertexts)."""
        return _native.MulPlaintextAccumNew(
            [int(c) for c in ct_ids], [int(p) for p in pt_ids])

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

    def _rotate_lt(self, ct_id: int, k: int) -> int:
        """Rotate by k for linear-transform evaluation. RotateNew lazily
        generates a dedicated key for this exact distance; cross-layer key
        reuse comes from ORION_CHEDDAR_SHARED_BS (see __init__).

        A POW2_ROTATE mode used to live here (decompose k into a chain of
        rotations over a fixed universal power-of-two key set, trading up
        to log2(slots) HRots per logical rotation for a tiny key set).
        Removed after the 2026-07-29 runtime ablation: it had no runtime
        benefit in any combination (28.9-29.7s, same band as exact-key),
        and its memory advantage is moot since MANAGED_MEMORY is required
        on this 24GB card either way -- exact-key + SHARED_BS=64 +
        BOOT_MIN_KS=1 is both faster and small enough (41GB peak).
        """
        if k == 0:
            return ct_id
        return self.RotateNew(ct_id, k)

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
        # values_per_diag is this transform's own authoritative slot
        # width -- whatever packing.diagonalize() actually built each
        # diagonal at (self.slots on the layer, propagated per-layer for
        # cheddar; see packing.propagate_physical_slots). It used to
        # always equal self._slots (the scheme's fixed global max) back
        # when every layer packed at the global width unconditionally,
        # which made the old `pad/truncate to self._slots` below a
        # silent no-op. Now that layers can pack narrower, doing that
        # padding would re-widen every diagonal back to the global max
        # right before encoding it -- defeating narrow packing at the
        # one place nothing else could catch it (each diagonal is still
        # numerically correct zero-padded, so a layer evaluated in
        # isolation looks fine; only a downstream layer, packed for the
        # narrower width, then receives a wider-than-expected ciphertext
        # and silently mis-wraps its own rotations).
        values_per_diag = len(diags_data) // num_diags
        diags: dict[int, list[float]] = {}
        for i, idx in enumerate(diags_idxs):
            start = i * values_per_diag
            diags[int(idx)] = list(diags_data[start:start + values_per_diag])

        use_bsgs = bsgs_ratio not in ("none", None) and len(diags) > 1
        if use_bsgs:
            bs = (self._shared_bs if self._shared_bs is not None else
                  max(1, math.ceil(math.sqrt(len(diags) * float(bsgs_ratio)))))
        else:
            bs = None

        # Double-hoisted path: hand the raw diagonals to the native
        # HoistHandler (it pre-encodes its own per-(gs,bs) plaintexts with
        # the giant-step roll baked in, so the Python-side pt cache below
        # is skipped entirely). ALL blocks route through it when enabled --
        # mixing native (pre-rescaled) and Python (unrescaled) blocks in
        # one accumulation row would mismatch scales.
        if self._double_hoist:
            hoist_bs = bs if bs is not None else max(
                1, math.ceil(math.sqrt(max(1, len(diags)))))
            hoist_id = _native.GenerateHoistLT(
                [int(k) for k in diags],
                [float(v) for vals in diags.values() for v in vals],
                int(level), int(hoist_bs))
            lt_id = self._next_lt_id
            self._next_lt_id += 1
            self._transforms[lt_id] = {"hoist_id": hoist_id}
            return lt_id

        # Encode every diagonal's plaintext once, here at compile time,
        # instead of on every EvaluateLinearTransform call. Diagonal values
        # are the model's fixed trained weights -- they never depend on the
        # input ciphertext -- so re-encoding them per-inference was pure
        # waste. Profiled on ResNet20: ~25-29% of total runtime was spent
        # right here (cheddar::Encoder::Encode/EncodeWorker), with the GPU
        # sitting idle the whole time (CPU-bound encode). BSGS's giant-step
        # roll is diagonal-index-determined too, so it's baked into the
        # cached plaintext rather than redone per evaluate.
        pt_ids: dict[int, int] = {}
        if use_bsgs:
            for k, vals in diags.items():
                g = k // bs
                shifted = vals if g == 0 else np.roll(vals, g * bs).tolist()
                pt_ids[k] = self._lt_encode(shifted, level)
        else:
            for k, vals in diags.items():
                pt_ids[k] = self._lt_encode(vals, level)

        lt_id = self._next_lt_id
        self._next_lt_id += 1
        self._transforms[lt_id] = {
            "diags": diags, "bsgs_ratio": bsgs_ratio,
            "use_bsgs": use_bsgs, "bs": bs, "pt_ids": pt_ids,
        }
        return lt_id

    def _lt_use_bsgs(self, lt_id: int) -> bool:
        return self._transforms[lt_id]["use_bsgs"]

    def _lt_bsgs_stride(self, lt_id: int) -> int:
        return self._transforms[lt_id]["bs"]

    def GetLinearTransformRotationKeys(self, lt_id: int) -> list[int]:
        t = self._transforms[lt_id]
        if "hoist_id" in t:
            # Ask the constructed handler -- it may have swapped bs/gs
            # internally (single-giant-group optimization), changing
            # which distances it needs. Keys land in the shared
            # max-level map via the normal AddRotationKey flow; the
            # fused kernels handle key level >= input level.
            return list(_native.GetHoistLTRotationKeys(t["hoist_id"]))
        diags = t["diags"]
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
        t = self._transforms[lt_id]
        if "hoist_id" in t:
            return _native.EvaluateHoistLT(t["hoist_id"], int(ct_id))
        if t["use_bsgs"]:
            return self._eval_lt_bsgs(ct_id, t["diags"], t["bs"], t["pt_ids"])
        return self._eval_lt_naive(ct_id, t["diags"], t["pt_ids"])

    def _lt_encode(self, values, level: int) -> int:
        return self.Encode(values, level, self._default_scale)

    def _rotate_lt_batch(self, ct_id: int, ks: list[int]) -> dict[int, int]:
        """Rotate one ciphertext by every distance in ks (nonzero, unique),
        as a single hoisted batch. All these rotations share the same
        input, so RotateBatchNew (hoist_mode="single") pays the ModUp once
        for the whole batch -- measured 1.5-1.6x over an equivalent solo
        RotateNew loop at real baby-step batch sizes (n=8..63). Caller
        owns (and must delete) the returned ciphertexts."""
        if len(ks) < 2:
            return {k: self.RotateNew(ct_id, k) for k in ks}
        return dict(zip(ks, self.RotateBatchNew(ct_id, ks)))

    def _eval_lt_naive(self, ct_id: int, diags: dict[int, list[float]],
                       pt_ids: dict[int, int]) -> int:
        # Every diagonal rotates the same input ct -- hoist them as one
        # batch (k=0 uses the input directly), then fold the whole
        # rotate*diag sum into one fused MAC call.
        rots = self._rotate_lt_batch(ct_id, [k for k in diags if k != 0])
        ks = list(diags)
        result = self.MulPlaintextAccumNew(
            [rots.get(k, ct_id) for k in ks], [pt_ids[k] for k in ks])
        for rot in rots.values():
            self.DeleteCiphertext(rot)
        return result

    def _eval_lt_bsgs(self, ct_id: int, diags: dict[int, list[float]],
                      bs: int, pt_ids: dict[int, int]) -> int:
        giant_groups: dict[int, list[tuple[int, int]]] = {}
        for k in diags:
            b, g = k % bs, k // bs
            giant_groups.setdefault(g, []).append((b, k))

        # Baby-step rotations all share the input ct -> one hoisted batch.
        # Giant-step rotations (below) each apply to a different
        # accumulated ciphertext, so they can't join the batch.
        needed_babies = sorted({b for group in giant_groups.values()
                                for b, _ in group})
        baby_rots = self._rotate_lt_batch(
            ct_id, [b for b in needed_babies if b != 0])
        if needed_babies and needed_babies[0] == 0:
            baby_rots[0] = ct_id

        result = None
        for g, group in giant_groups.items():
            # One fused MAC per giant group: sum_b baby_rots[b] * diag_k
            # replaces len(group) mults + len(group)-1 adds.
            inner = self.MulPlaintextAccumNew(
                [baby_rots[b] for b, _ in group],
                [pt_ids[k] for _, k in group])

            partial = self._rotate_lt(inner, g * bs)
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
        t = self._transforms.pop(lt_id, None)
        if t is None:
            return
        if "hoist_id" in t:
            _native.DeleteHoistLT(t["hoist_id"])
            return
        for pt_id in t["pt_ids"].values():
            self.DeletePlaintext(pt_id)

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
        # EvalPoly (native, log-depth) requires degree >= 2; below that
        # Horner's method is already both cheap and exact.
        if len(coeffs) - 1 >= 2:
            return _native.EvaluatePolynomialNative(
                int(ct_id), coeffs, float(scale))
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

        # Each stage's fit interval is inflated beyond the previous
        # stage's CLEAR output range: the actual FHE input to stage i+1
        # is stage i's output plus accumulated ciphertext noise, and a
        # least-squares Chebyshev fit is completely unconstrained
        # outside its fitted interval -- a degree-27 tail can hit +-75
        # within 10% past the edge. Measured on the 32-bit chain
        # (word32 seed 5, ResNet20 layers_1_0_act1): stage-1 clear
        # output +-1.0x, FHE input to stage 2 reached +-1.103 -> stage-2
        # output exploded to [-57, +75] and corrupted the whole
        # inference, deterministically per noise draw (~40% of draws).
        # Fitting over 1.25x the clear range keeps every stage
        # sign-like across the entire noise-reachable interval; costs a
        # slightly coarser fit of the same degree (ResNet20 MAE 0.0003
        # -> 0.0006), nothing at runtime. Applied in word32 mode only:
        # 64-bit noise never approaches the cliff (no corruption in
        # dozens of runs), so its fit -- and its documented benchmark
        # numbers -- stay bit-identical.
        noise_domain_margin = 1.25 if _WORD32 else 1.0

        coeffs_flat: list[float] = []
        for i, deg in enumerate(degrees):
            is_last = (i == len(degrees) - 1)
            n_pts = max(8 * deg, 500)

            fit_min = domain_min * noise_domain_margin
            fit_max = domain_max * noise_domain_margin
            x_neg = np.linspace(fit_min, -current_gap, n_pts)
            x_pos = np.linspace(current_gap, fit_max, n_pts)
            x = np.concatenate([x_neg, x_pos])
            y = np.where(x > 0, 1.0, 0.0) if is_last else np.sign(x)

            coeffs = self._fit_chebyshev_standard_basis(x, y, deg)
            coeffs_flat.extend(coeffs[:deg + 1].tolist())

            if not is_last:
                # Next stage's domain = this stage's output over the
                # whole noise-reachable (inflated) input interval.
                dense_neg = np.linspace(fit_min, -current_gap, 2000)
                dense_pos = np.linspace(current_gap, fit_max, 2000)
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

    def GetPeakDeviceMemoryMB(self) -> float:
        """Peak bytes ever allocated through the scheme's RMM pool -- keys,
        ciphertexts, everything, not just evaluation keys. See
        MemoryPool.h; measures the true footprint even when
        ORION_CHEDDAR_MANAGED_MEMORY pages part of it to host RAM."""
        return _native.GetPeakDeviceMemoryMB()

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
            f"lattigo backend, or extend the wrap in "
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
