"""DeSiLo FHE backend — drop-in replacement for the Lattigo backend.

Exposes the same ID-based interface as LattigoLibrary so the Python
wrapper layer (orion/backend/python/) works unchanged.
"""

import numpy as np
import desilofhe


class DeSiLoLibrary:
    """DeSiLo FHE backend implementing the LattigoLibrary interface."""

    def __init__(self):
        self.engine = None

        # ID-based object registry (mirrors Lattigo's integer-ID approach)
        self._objects = {}      # id -> native DeSiLo object
        self._next_id = 1
        self._scales = {}       # id -> scale metadata (DeSiLo doesn't expose scale)
        self._obj_type = {}     # id -> "pt" | "ct" (for GetLive* queries)

        # Keys (stored internally, passed explicitly to DeSiLo operations)
        self._sk = None
        self._pk = None
        self._rk = None
        self._rot_key = None
        self._conj_key = None

        # Scheme params
        self._default_scale = None
        self._max_level = None
        self._slots = None

        # Polynomial storage: poly_id -> (coeffs_ascending, type)
        self._polynomials = {}

        # Linear transform storage: lt_id -> dict with diag data
        self._transforms = {}

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _store(self, obj, kind, scale=None):
        """Store a DeSiLo object, return integer ID."""
        oid = self._next_id
        self._next_id += 1
        self._objects[oid] = obj
        self._obj_type[oid] = kind
        if scale is not None:
            self._scales[oid] = scale
        return oid

    def _get(self, oid):
        return self._objects[oid]

    def _delete(self, oid):
        self._objects.pop(oid, None)
        self._scales.pop(oid, None)
        self._obj_type.pop(oid, None)

    def _replace(self, oid, new_obj):
        """Replace object at existing ID (for in-place operations)."""
        self._objects[oid] = new_obj
        return oid

    # ------------------------------------------------------------------
    #  setup_bindings  (called by Scheme.setup_backend)
    # ------------------------------------------------------------------

    def setup_bindings(self, orion_params):
        self.setup_scheme(orion_params)
        self.setup_tensor_binds()
        self.setup_key_generator()
        self.setup_encoder()
        self.setup_encryptor()
        self.setup_evaluator()
        self.setup_poly_evaluator()
        self.setup_lt_evaluator()
        self.setup_bootstrapper()

    # ------------------------------------------------------------------
    #  Scheme setup
    # ------------------------------------------------------------------

    def setup_scheme(self, orion_params):
        logq = orion_params.get_logq()
        logscale = orion_params.get_logscale()

        self._max_level = len(logq) - 1
        self._default_scale = 1 << logscale
        self._slots = orion_params.get_slots()

        # Detect bootstrap requirement: boot_logp differs from logp when
        # the config explicitly provides boot_params.
        boot_logp = orion_params.get_boot_logp()
        logp = orion_params.get_logp()
        self._use_bootstrap = (boot_logp != logp)
        self._device = orion_params.get_device()

        if self._use_bootstrap:
            print(f"[DeSiLo] Creating BOOTSTRAP engine: "
                  f"slot_count={self._slots}, device={self._device}")
            self.engine = desilofhe.Engine(
                use_bootstrap=True, slot_count=self._slots, mode=self._device)
        else:
            print(f"[DeSiLo] Creating engine: target max_level="
                  f"{self._max_level}, slots={self._slots}, device={self._device}")
            self.engine = desilofhe.Engine(self._max_level, mode=self._device)

        print(f"[DeSiLo] Engine ready: max_level={self.engine.max_level}, "
              f"slot_count={self.engine.slot_count}")

        if self.engine.slot_count != self._slots:
            print(f"[DeSiLo] WARNING: slot_count mismatch — "
                  f"engine={self.engine.slot_count}, expected={self._slots}")

    def DeleteScheme(self):
        self._objects.clear()
        self._scales.clear()
        self._obj_type.clear()
        self._polynomials.clear()
        self._transforms.clear()
        self._sk = None
        self._pk = None
        self._rk = None
        self._rot_key = None
        self._conj_key = None
        self.engine = None

    def FreeCArray(self, ptr=None):
        pass  # No C memory to free in DeSiLo

    # ------------------------------------------------------------------
    #  Tensor metadata binds
    # ------------------------------------------------------------------

    def setup_tensor_binds(self):
        pass  # Methods defined directly on this class

    def DeletePlaintext(self, pt_id):
        self._delete(pt_id)

    def DeleteCiphertext(self, ct_id):
        self._delete(ct_id)

    def GetPlaintextScale(self, pt_id):
        return self._scales.get(pt_id, self._default_scale)

    def GetCiphertextScale(self, ct_id):
        return self._scales.get(ct_id, self._default_scale)

    def SetPlaintextScale(self, pt_id, scale):
        self._scales[pt_id] = scale

    def SetCiphertextScale(self, ct_id, scale):
        self._scales[ct_id] = scale

    def GetPlaintextLevel(self, pt_id):
        return self._get(pt_id).level

    def GetCiphertextLevel(self, ct_id):
        return self._get(ct_id).level

    def GetPlaintextSlots(self, pt_id):
        return self.engine.slot_count

    def GetCiphertextSlots(self, ct_id):
        return self.engine.slot_count

    def GetCiphertextDegree(self, ct_id):
        return 1  # Always relinearized in our usage

    def GetModuliChain(self):
        # DeSiLo manages scale internally, so the exact prime values don't
        # affect computation. We return a list of the default scale repeated
        # for each level so that callers (bootstrap, batch norm, composite
        # activations) can index by level without crashing.
        # This is a compatibility stub — redundant if we drop Lattigo support
        # and refactor the wrapper layer to stop querying the moduli chain.
        return [self._default_scale] * (self.engine.max_level + 1)

    def GetAuxModuliChain(self):
        return []

    def GetLivePlaintexts(self):
        return [k for k, v in self._obj_type.items() if v == "pt"]

    def GetLiveCiphertexts(self):
        return [k for k, v in self._obj_type.items() if v == "ct"]

    # ------------------------------------------------------------------
    #  Key generation
    # ------------------------------------------------------------------

    def setup_key_generator(self):
        pass

    def NewKeyGenerator(self):
        pass  # Engine handles key generation directly

    def GenerateSecretKey(self):
        self._sk = self.engine.create_secret_key()
        print(f"[DeSiLo] Secret key generated")

    def GeneratePublicKey(self):
        self._pk = self.engine.create_public_key(self._sk)
        print(f"[DeSiLo] Public key generated")

    def GenerateRelinearizationKey(self):
        self._rk = self.engine.create_relinearization_key(self._sk)
        print(f"[DeSiLo] Relinearization key generated")

    def GenerateEvaluationKeys(self):
        self._rot_key = self.engine.create_rotation_key(self._sk)
        self._conj_key = self.engine.create_conjugation_key(self._sk)
        print(f"[DeSiLo] Evaluation keys generated (rotation + conjugation)")

    def SerializeSecretKey(self):
        data = self.engine.serialize_secret_key(self._sk)
        return np.frombuffer(data, dtype=np.uint8), None

    def LoadSecretKey(self, byte_data):
        self._sk = self.engine.deserialize_secret_key(bytes(byte_data))

    # ------------------------------------------------------------------
    #  Encode / Decode
    # ------------------------------------------------------------------

    def setup_encoder(self):
        pass

    def NewEncoder(self):
        pass

    def Encode(self, values, level, scale):
        pt = self.engine.encode(values, level=level)
        return self._store(pt, "pt", scale=scale)

    def Decode(self, pt_id):
        pt = self._get(pt_id)
        result = self.engine.decode(pt)
        return result.real.tolist()

    # ------------------------------------------------------------------
    #  Encrypt / Decrypt
    # ------------------------------------------------------------------

    def setup_encryptor(self):
        pass

    def NewEncryptor(self):
        pass

    def NewDecryptor(self):
        pass

    def Encrypt(self, pt_id):
        pt = self._get(pt_id)
        ct = self.engine.encrypt(pt, self._pk)
        scale = self._scales.get(pt_id, self._default_scale)
        return self._store(ct, "ct", scale=scale)

    def Decrypt(self, ct_id):
        ct = self._get(ct_id)
        pt = self.engine.decrypt_to_plaintext(ct, self._sk)
        scale = self._scales.get(ct_id, self._default_scale)
        return self._store(pt, "pt", scale=scale)

    # ------------------------------------------------------------------
    #  Evaluator setup
    # ------------------------------------------------------------------

    def setup_evaluator(self):
        pass

    def NewEvaluator(self):
        pass

    # ------------------------------------------------------------------
    #  Rotation
    # ------------------------------------------------------------------

    def AddRotationKey(self, k):
        pass  # General rotation key already covers all deltas

    def Rotate(self, ct_id, k):
        ct = self._get(ct_id)
        # CRITICAL: negate k — Lattigo positive k = left-shift,
        # DeSiLo positive delta = right-shift
        result = self.engine.rotate(ct, self._rot_key, delta=-k)
        return self._replace(ct_id, result)

    def RotateNew(self, ct_id, k):
        ct = self._get(ct_id)
        result = self.engine.rotate(ct, self._rot_key, delta=-k)
        scale = self._scales.get(ct_id, self._default_scale)
        return self._store(result, "ct", scale=scale)

    # ------------------------------------------------------------------
    #  Negate
    # ------------------------------------------------------------------

    def Negate(self, ct_id):
        ct = self._get(ct_id)
        result = self.engine.negate(ct)
        scale = self._scales.get(ct_id, self._default_scale)
        return self._store(result, "ct", scale=scale)

    # ------------------------------------------------------------------
    #  Rescale  (no-op — DeSiLo auto-rescales during multiply)
    # ------------------------------------------------------------------

    def Rescale(self, ct_id):
        return ct_id

    def RescaleNew(self, ct_id):
        ct = self._get(ct_id)
        result = self.engine.clone(ct)
        scale = self._scales.get(ct_id, self._default_scale)
        return self._store(result, "ct", scale=scale)

    # ------------------------------------------------------------------
    #  Scalar arithmetic
    # ------------------------------------------------------------------

    def AddScalar(self, ct_id, scalar):
        ct = self._get(ct_id)
        result = self.engine.add(ct, float(scalar))
        return self._replace(ct_id, result)

    def AddScalarNew(self, ct_id, scalar):
        ct = self._get(ct_id)
        result = self.engine.add(ct, float(scalar))
        scale = self._scales.get(ct_id, self._default_scale)
        return self._store(result, "ct", scale=scale)

    def SubScalar(self, ct_id, scalar):
        ct = self._get(ct_id)
        result = self.engine.subtract(ct, float(scalar))
        return self._replace(ct_id, result)

    def SubScalarNew(self, ct_id, scalar):
        ct = self._get(ct_id)
        result = self.engine.subtract(ct, float(scalar))
        scale = self._scales.get(ct_id, self._default_scale)
        return self._store(result, "ct", scale=scale)

    def MulScalarInt(self, ct_id, scalar):
        ct = self._get(ct_id)
        result = self.engine.multiply(ct, int(scalar))
        return self._replace(ct_id, result)

    def MulScalarIntNew(self, ct_id, scalar):
        ct = self._get(ct_id)
        result = self.engine.multiply(ct, int(scalar))
        scale = self._scales.get(ct_id, self._default_scale)
        return self._store(result, "ct", scale=scale)

    def MulScalarFloat(self, ct_id, scalar):
        ct = self._get(ct_id)
        result = self.engine.multiply(ct, float(scalar))
        return self._replace(ct_id, result)

    def MulScalarFloatNew(self, ct_id, scalar):
        ct = self._get(ct_id)
        result = self.engine.multiply(ct, float(scalar))
        scale = self._scales.get(ct_id, self._default_scale)
        return self._store(result, "ct", scale=scale)

    # ------------------------------------------------------------------
    #  Plaintext arithmetic
    # ------------------------------------------------------------------

    def AddPlaintext(self, ct_id, pt_id):
        ct, pt = self._get(ct_id), self._get(pt_id)
        result = self.engine.add(ct, pt)
        return self._replace(ct_id, result)

    def AddPlaintextNew(self, ct_id, pt_id):
        ct, pt = self._get(ct_id), self._get(pt_id)
        result = self.engine.add(ct, pt)
        scale = self._scales.get(ct_id, self._default_scale)
        return self._store(result, "ct", scale=scale)

    def SubPlaintext(self, ct_id, pt_id):
        ct, pt = self._get(ct_id), self._get(pt_id)
        result = self.engine.subtract(ct, pt)
        return self._replace(ct_id, result)

    def SubPlaintextNew(self, ct_id, pt_id):
        ct, pt = self._get(ct_id), self._get(pt_id)
        result = self.engine.subtract(ct, pt)
        scale = self._scales.get(ct_id, self._default_scale)
        return self._store(result, "ct", scale=scale)

    def MulPlaintext(self, ct_id, pt_id):
        ct, pt = self._get(ct_id), self._get(pt_id)
        result = self.engine.multiply(ct, pt)
        return self._replace(ct_id, result)

    def MulPlaintextNew(self, ct_id, pt_id):
        ct, pt = self._get(ct_id), self._get(pt_id)
        result = self.engine.multiply(ct, pt)
        scale = self._scales.get(ct_id, self._default_scale)
        return self._store(result, "ct", scale=scale)

    # ------------------------------------------------------------------
    #  Ciphertext arithmetic
    # ------------------------------------------------------------------

    def AddCiphertext(self, ct_id1, ct_id2):
        ct1, ct2 = self._get(ct_id1), self._get(ct_id2)
        result = self.engine.add(ct1, ct2)
        return self._replace(ct_id1, result)

    def AddCiphertextNew(self, ct_id1, ct_id2):
        ct1, ct2 = self._get(ct_id1), self._get(ct_id2)
        result = self.engine.add(ct1, ct2)
        scale = self._scales.get(ct_id1, self._default_scale)
        return self._store(result, "ct", scale=scale)

    def SubCiphertext(self, ct_id1, ct_id2):
        ct1, ct2 = self._get(ct_id1), self._get(ct_id2)
        result = self.engine.subtract(ct1, ct2)
        return self._replace(ct_id1, result)

    def SubCiphertextNew(self, ct_id1, ct_id2):
        ct1, ct2 = self._get(ct_id1), self._get(ct_id2)
        result = self.engine.subtract(ct1, ct2)
        scale = self._scales.get(ct_id1, self._default_scale)
        return self._store(result, "ct", scale=scale)

    def MulRelinCiphertext(self, ct_id1, ct_id2):
        ct1, ct2 = self._get(ct_id1), self._get(ct_id2)
        result = self.engine.multiply(ct1, ct2, self._rk)
        return self._replace(ct_id1, result)

    def MulRelinCiphertextNew(self, ct_id1, ct_id2):
        ct1, ct2 = self._get(ct_id1), self._get(ct_id2)
        result = self.engine.multiply(ct1, ct2, self._rk)
        scale = self._scales.get(ct_id1, self._default_scale)
        return self._store(result, "ct", scale=scale)

    # ------------------------------------------------------------------
    #  Polynomial evaluation
    # ------------------------------------------------------------------

    def setup_poly_evaluator(self):
        pass

    def NewPolynomialEvaluator(self):
        pass

    def GenerateMonomial(self, coeffs):
        """Store monomial coefficients. The wrapper already reversed them
        to ascending order [a0, a1, ..., an] before calling this."""
        poly_id = self._next_id
        self._next_id += 1
        self._polynomials[poly_id] = (list(coeffs), "monomial")
        return poly_id

    def GenerateChebyshev(self, coeffs):
        """Store Chebyshev coefficients [a0, a1, ..., an]."""
        poly_id = self._next_id
        self._next_id += 1
        self._polynomials[poly_id] = (list(coeffs), "chebyshev")
        return poly_id

    def EvaluatePolynomial(self, ct_id, poly_id, scale):
        ct = self._get(ct_id)
        coeffs, poly_type = self._polynomials[poly_id]

        if poly_type == "monomial":
            result = self.engine.evaluate_polynomial(ct, coeffs, self._rk)
        else:  # chebyshev
            # DeSiLo rejects Chebyshev polys when all non-constant
            # coefficients have |value| < 1.0.  Fall back to monomial
            # basis (mathematically equivalent, same level consumption).
            has_large_nonconst = any(abs(c) >= 1.0 for c in coeffs[1:])
            if has_large_nonconst:
                result = self.engine.evaluate_chebyshev_polynomial(
                    ct, coeffs, self._rk)
            else:
                from numpy.polynomial.chebyshev import cheb2poly
                mono_coeffs = cheb2poly(coeffs).tolist()
                result = self.engine.evaluate_polynomial(
                    ct, mono_coeffs, self._rk)

        return self._store(result, "ct", scale=self._default_scale)

    @staticmethod
    def _fit_chebyshev_standard_basis(x, y, degree):
        """Least-squares fit in the standard Chebyshev basis T_n(x).

        Unlike numpy's Chebyshev.fit this performs NO domain mapping,
        so the returned coefficients c satisfy:
            sum_j c[j] * T_j(x) ≈ y
        for the actual x values provided.
        """
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
        """Evaluate polynomial in standard Chebyshev basis."""
        n = len(coeffs)
        T = np.zeros((len(x), n))
        T[:, 0] = 1.0
        if n > 1:
            T[:, 1] = x
        for j in range(2, n):
            T[:, j] = 2.0 * x * T[:, j - 1] - T[:, j - 2]
        return T @ coeffs

    def GenerateMinimaxSignCoeffs(self, degrees, prec, logalpha, logerr, debug):
        """Generate composite sign polynomial coefficients.

        Each stage is fitted on the *output range* of the previous stage
        (matching Lattigo's composite Remez strategy) so that the
        composition converges towards sign(x).

        The first n-1 polynomials approximate sign(x) in {-1, 1}.
        The last polynomial approximates step(x) in {0, 1}, matching
        Lattigo's convention (used by _Sign / ReLU: x * step(x)).
        """
        gap = 2.0 ** (-logalpha)
        domain_min, domain_max = -1.0, 1.0
        current_gap = gap

        coeffs_flat = []
        for i, deg in enumerate(degrees):
            is_last = (i == len(degrees) - 1)
            n_pts = max(8 * deg, 500)

            x_neg = np.linspace(domain_min, -current_gap, n_pts)
            x_pos = np.linspace(current_gap, domain_max, n_pts)
            x = np.concatenate([x_neg, x_pos])

            if is_last:
                y = np.where(x > 0, 1.0, 0.0)
            else:
                y = np.sign(x)

            coeffs = self._fit_chebyshev_standard_basis(x, y, deg)
            coeffs_flat.extend(coeffs[:deg + 1].tolist())

            if not is_last:
                # Determine output range for the next stage
                dense_neg = np.linspace(domain_min, -current_gap, 2000)
                dense_pos = np.linspace(current_gap, domain_max, 2000)
                dense = np.concatenate([dense_neg, dense_pos])
                output = self._eval_chebyshev_standard_basis(dense, coeffs)

                domain_min = float(output.min())
                domain_max = float(output.max())

                # New gap: the polynomial's value at the old gap boundary
                near_gap = np.array([current_gap, -current_gap])
                near_out = self._eval_chebyshev_standard_basis(near_gap, coeffs)
                current_gap = float(min(abs(near_out[0]), abs(near_out[1])))
                # Ensure gap doesn't collapse to zero
                current_gap = max(current_gap, 1e-12)

        return coeffs_flat

    # ------------------------------------------------------------------
    #  Linear transform
    # ------------------------------------------------------------------

    def setup_lt_evaluator(self):
        pass

    def NewLinearTransformEvaluator(self):
        pass

    def GenerateLinearTransform(self, diags_idxs, diags_data, level,
                                bsgs_ratio, io_mode):
        """Store diagonal data as numpy arrays for later evaluation."""
        slots = self.engine.slot_count
        num_diags = len(diags_idxs)
        values_per_diag = len(diags_data) // num_diags

        diags = {}
        for i, idx in enumerate(diags_idxs):
            start = i * values_per_diag
            end = start + values_per_diag
            diag_vals = diags_data[start:end]
            if len(diag_vals) < slots:
                diag_vals = diag_vals + [0.0] * (slots - len(diag_vals))
            diags[idx] = np.array(diag_vals[:slots], dtype=np.float64)

        lt_id = self._next_id
        self._next_id += 1
        self._transforms[lt_id] = {
            "diags": diags,
            "level": level,
            "bsgs_ratio": bsgs_ratio,
        }
        return lt_id

    def EvaluateLinearTransform(self, lt_id, ct_id):
        """Evaluate linear transform via diagonal rotation method.

        For each diagonal k with values d_k:
            result[i] += d_k[i] * input[(i+k) % n]

        Uses BSGS (Baby-step Giant-step) when bsgs_ratio is set, reducing
        rotations from O(N) to O(2*sqrt(N)). Falls back to naive loop when
        bsgs_ratio is "none" (e.g. oracle tests).
        """
        ct = self._get(ct_id)
        transform = self._transforms[lt_id]
        diags = transform["diags"]
        bsgs_ratio = transform["bsgs_ratio"]

        use_bsgs = (bsgs_ratio not in ("none", None)
                     and len(diags) > 1)

        if use_bsgs:
            result = self._eval_lt_bsgs(ct, diags, int(bsgs_ratio))
        else:
            result = self._eval_lt_naive(ct, diags)

        return self._store(result, "ct", scale=self._default_scale)

    def _eval_lt_naive(self, ct, diags):
        """Naive diagonal loop using individual rotate calls."""
        ct_level = ct.level
        result = None
        for diag_idx, diag_arr in diags.items():
            # Rotate ciphertext (negate: Lattigo left-shift vs DeSiLo right-shift)
            if diag_idx == 0:
                ct_rot = self.engine.clone(ct)
            else:
                ct_rot = self.engine.rotate(
                    ct, self._rot_key, delta=-diag_idx)

            # Encode diagonal and multiply
            pt_diag = self.engine.encode(diag_arr, level=ct_level)
            prod = self.engine.multiply(ct_rot, pt_diag)

            if result is None:
                result = prod
            else:
                result = self.engine.add(result, prod)
        return result

    def _eval_lt_bsgs(self, ct, diags, bsgs_ratio):
        """BSGS-optimized diagonal evaluation.

        Decomposes each diagonal index k = b + g*bs (baby + giant*stride).
        Precomputes bs baby-step rotations, then for each giant step
        accumulates inner products and applies one giant-step rotation.
        Reduces total rotations from O(N) to O(2*sqrt(N)).

        Uses individual engine.rotate() calls instead of rotate_batch()
        to avoid segfaults with large numbers of rotations.
        """
        import math
        num_diags = len(diags)
        bs = max(1, math.ceil(math.sqrt(num_diags * bsgs_ratio)))
        ct_level = ct.level

        # Group diagonals by giant step
        giant_groups = {}  # g -> list of (b, k, arr)
        for k, arr in diags.items():
            b = k % bs
            g = k // bs
            giant_groups.setdefault(g, []).append((b, k, arr))

        # Precompute baby-step rotations using individual rotate calls
        needed_babies = sorted({b for group in giant_groups.values()
                                for b, _, _ in group})
        baby_rots = {}
        for b in needed_babies:
            if b == 0:
                baby_rots[0] = self.engine.clone(ct)
            else:
                baby_rots[b] = self.engine.rotate(
                    ct, self._rot_key, delta=-b)

        # Accumulate over giant steps
        result = None
        for g, group in giant_groups.items():
            inner = None
            for b, k, arr in group:
                # Pre-rotate plaintext values to compensate for
                # the giant-step rotation applied below.
                if g != 0:
                    shifted = np.roll(arr, g * bs)
                else:
                    shifted = arr

                # Encode the diagonal values and multiply
                pt_diag = self.engine.encode(shifted, level=ct_level)
                prod = self.engine.multiply(baby_rots[b], pt_diag)

                if inner is None:
                    inner = prod
                else:
                    inner = self.engine.add(inner, prod)

            # Apply giant-step rotation (negate for direction convention)
            if g != 0:
                inner = self.engine.rotate(
                    inner, self._rot_key, delta=-(g * bs))

            if result is None:
                result = inner
            else:
                result = self.engine.add(result, inner)

        return result

    def DeleteLinearTransform(self, lt_id):
        self._transforms.pop(lt_id, None)

    def GetLinearTransformRotationKeys(self, lt_id):
        # DeSiLo uses general rotation key; return empty list
        return []

    def GenerateLinearTransformRotationKey(self, k):
        pass  # General rotation key covers all

    def GenerateAndSerializeRotationKey(self, k):
        # Return dummy data — DeSiLo uses general rotation key
        data = self.engine.serialize_rotation_key(self._rot_key)
        return np.frombuffer(data, dtype=np.uint8), None

    def LoadRotationKey(self, byte_data, k=None):
        pass  # General rotation key already loaded

    def SerializeDiagonal(self, lt_id, diag_idx):
        # Stub for I/O mode serialization
        return np.array([], dtype=np.uint8), None

    def LoadPlaintextDiagonal(self, byte_data, lt_id, diag_idx):
        pass

    def RemovePlaintextDiagonals(self, lt_id):
        pass

    def RemoveRotationKeys(self):
        pass

    # ------------------------------------------------------------------
    #  Bootstrap
    # ------------------------------------------------------------------

    def setup_bootstrapper(self):
        self._boot_key = None

    def NewBootstrapper(self, logPs, slots):
        """Create a bootstrap key for the given slot count.

        DeSiLo's bootstrap uses a single BootstrapKey created from the
        secret key.  The logPs parameter (Lattigo-specific auxiliary prime
        sizes) is ignored — DeSiLo manages its own auxiliary primes.
        """
        if self._sk is None:
            raise RuntimeError(
                "[DeSiLo] Cannot create bootstrap key: secret key not "
                "generated yet.")

        print(f"[DeSiLo] Creating bootstrap key for slots={slots} ...")
        # Free previous bootstrap key to reclaim GPU memory before
        # allocating a new one (each key is ~12GB on GPU).
        if self._boot_key is not None:
            del self._boot_key
            self._boot_key = None
            import gc; gc.collect()
        self._boot_key = self.engine.create_bootstrap_key(self._sk)
        print(f"[DeSiLo] Bootstrap key ready.")

    def Bootstrap(self, ct_id, slots):
        ct = self._get(ct_id)
        if self._boot_key is None:
            raise RuntimeError(
                "[DeSiLo] No bootstrap key. Call NewBootstrapper first.")

        # Caller (nn.operations.Bootstrap) is required to prescale values into
        # [-1, 1] and drive the ciphertext to level 0 before invoking this.
        ct = self.engine.intt(ct)
        result = self.engine.bootstrap(
            ct, self._rk, self._conj_key, self._boot_key)

        scale = self._scales.get(ct_id, self._default_scale)
        return self._store(result, "ct", scale=scale)

    def DeleteBootstrappers(self):
        self._boot_key = None
