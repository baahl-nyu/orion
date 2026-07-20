"""Tests for BSGS-optimized linear transform evaluation.

These mirror the oracle linear transform tests but pass bsgs_ratio=2
to exercise the Baby-step Giant-step path in the DeSiLo backend.
"""

import sys
import os
import torch
import pytest

from orion.core.orion import Scheme

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "oracle"))
from fhe_test_utils import assert_fhe_close, encrypt_values


TEST_CKKS_CONFIG = {
    "ckks_params": {
        "LogN": 13,
        "LogQ": [29, 26, 26, 26],
        "LogP": [29],
        "LogScale": 26,
        "H": 8192,
        "RingType": "ConjugateInvariant",
    },
    "orion": {
        "backend": "desilo",
        "io_mode": "none",
        "debug": False,
    },
}

BSGS_RATIO = 2


@pytest.fixture(scope="module")
def scheme():
    s = Scheme()
    s.init_scheme(TEST_CKKS_CONFIG)
    yield s
    s.delete_scheme()


@pytest.fixture(scope="module")
def slots(scheme):
    return scheme.params.get_slots()


@pytest.fixture(scope="module")
def max_level(scheme):
    return scheme.params.get_max_level()


class TestBSGSLinearTransform:

    def test_identity_bsgs(self, scheme, slots, max_level):
        """Identity transform via BSGS should return input unchanged."""
        values = [float(i + 1) for i in range(slots)]
        level = max_level

        diags_idxs = [0]
        diags_data = [1.0] * slots

        lt_id = scheme.backend.GenerateLinearTransform(
            diags_idxs, diags_data, level, BSGS_RATIO, "none")

        ctxt = encrypt_values(scheme, values, level=level)
        ctxt_out_id = scheme.backend.EvaluateLinearTransform(
            lt_id, ctxt.ids[0])
        ctxt_out_id = scheme.backend.Rescale(ctxt_out_id)

        ptxt_out_id = scheme.backend.Decrypt(ctxt_out_id)
        result = scheme.backend.Decode(ptxt_out_id)

        assert_fhe_close(result, values, atol=1e-1,
                         msg="BSGS identity transform")
        scheme.backend.DeleteLinearTransform(lt_id)

    def test_scale_by_constant_bsgs(self, scheme, slots, max_level):
        """Scale transform via BSGS: result = c * input."""
        values = [float(i + 1) for i in range(slots)]
        c = 3.0
        level = max_level

        diags_idxs = [0]
        diags_data = [c] * slots

        lt_id = scheme.backend.GenerateLinearTransform(
            diags_idxs, diags_data, level, BSGS_RATIO, "none")

        ctxt = encrypt_values(scheme, values, level=level)
        ctxt_out_id = scheme.backend.EvaluateLinearTransform(
            lt_id, ctxt.ids[0])
        ctxt_out_id = scheme.backend.Rescale(ctxt_out_id)

        ptxt_out_id = scheme.backend.Decrypt(ctxt_out_id)
        result = scheme.backend.Decode(ptxt_out_id)

        expected = [c * v for v in values]
        assert_fhe_close(result, expected, atol=5e-1,
                         msg="BSGS scale transform")
        scheme.backend.DeleteLinearTransform(lt_id)

    def test_cyclic_permutation_bsgs(self, scheme, slots, max_level):
        """Cyclic shift via BSGS: single diagonal at index k."""
        values = [float(i + 1) for i in range(slots)]
        k = 3
        level = max_level

        diags_idxs = [k]
        diags_data = [1.0] * slots

        lt_id = scheme.backend.GenerateLinearTransform(
            diags_idxs, diags_data, level, BSGS_RATIO, "none")

        ctxt = encrypt_values(scheme, values, level=level)
        ctxt_out_id = scheme.backend.EvaluateLinearTransform(
            lt_id, ctxt.ids[0])
        ctxt_out_id = scheme.backend.Rescale(ctxt_out_id)

        ptxt_out_id = scheme.backend.Decrypt(ctxt_out_id)
        result = scheme.backend.Decode(ptxt_out_id)

        expected = [values[(i + k) % slots] for i in range(slots)]
        assert_fhe_close(result, expected, atol=1e-1,
                         msg=f"BSGS permute by {k}")
        scheme.backend.DeleteLinearTransform(lt_id)

    def test_multi_diagonal_bsgs(self, scheme, slots, max_level):
        """Two diagonals via BSGS: identity + shift-by-1."""
        values = [float(i + 1) for i in range(slots)]
        level = max_level

        diags_idxs = [0, 1]
        diags_data = [1.0] * slots + [1.0] * slots

        lt_id = scheme.backend.GenerateLinearTransform(
            diags_idxs, diags_data, level, BSGS_RATIO, "none")

        ctxt = encrypt_values(scheme, values, level=level)
        ctxt_out_id = scheme.backend.EvaluateLinearTransform(
            lt_id, ctxt.ids[0])
        ctxt_out_id = scheme.backend.Rescale(ctxt_out_id)

        ptxt_out_id = scheme.backend.Decrypt(ctxt_out_id)
        result = scheme.backend.Decode(ptxt_out_id)

        expected = [values[i] + values[(i + 1) % slots] for i in range(slots)]
        assert_fhe_close(result, expected, atol=5e-1,
                         msg="BSGS multi-diagonal")
        scheme.backend.DeleteLinearTransform(lt_id)

    def test_many_diagonals_bsgs(self, scheme, slots, max_level):
        """Many diagonals to properly exercise the BSGS baby/giant split.

        Uses 8 diagonals at varied offsets so that multiple giant steps
        are needed (with bs=ceil(sqrt(8*2))=4, we get giant steps 0,1,2).
        """
        values = [float(i + 1) for i in range(slots)]
        level = max_level

        # 8 diagonals at indices 0,1,2,3,5,7,9,11 with distinct weights
        diag_indices = [0, 1, 2, 3, 5, 7, 9, 11]
        weights = [0.5, -0.3, 0.2, 0.1, -0.4, 0.6, -0.2, 0.3]

        diags_data = []
        for w in weights:
            diags_data.extend([w] * slots)

        lt_id = scheme.backend.GenerateLinearTransform(
            diag_indices, diags_data, level, BSGS_RATIO, "none")

        ctxt = encrypt_values(scheme, values, level=level)
        ctxt_out_id = scheme.backend.EvaluateLinearTransform(
            lt_id, ctxt.ids[0])
        ctxt_out_id = scheme.backend.Rescale(ctxt_out_id)

        ptxt_out_id = scheme.backend.Decrypt(ctxt_out_id)
        result = scheme.backend.Decode(ptxt_out_id)

        # expected[i] = sum_j weight_j * input[(i + diag_j) % slots]
        expected = []
        for i in range(slots):
            val = sum(w * values[(i + d) % slots]
                      for d, w in zip(diag_indices, weights))
            expected.append(val)

        assert_fhe_close(result, expected, atol=5e-1,
                         msg="BSGS many-diagonals")
        scheme.backend.DeleteLinearTransform(lt_id)

    @pytest.mark.parametrize("num_diags", [50, 100, 144])
    def test_stress_many_diagonals_bsgs(self, scheme, slots, max_level,
                                         num_diags):
        """Stress test with large numbers of diagonals (50, 100, 144).

        144 diagonals matches the conv1 layer of ResNet that was causing
        segfaults with rotate_batch. Uses random but reproducible weights.
        """
        import numpy as np
        rng = np.random.default_rng(seed=42 + num_diags)

        values = [float(i % 17) * 0.1 for i in range(slots)]
        level = max_level

        # Spread diagonals across the slot range
        diag_indices = sorted(rng.choice(slots, size=num_diags,
                                         replace=False).tolist())
        weights = (rng.uniform(-0.5, 0.5, size=num_diags)).tolist()

        diags_data = []
        for w in weights:
            diags_data.extend([w] * slots)

        lt_id = scheme.backend.GenerateLinearTransform(
            diag_indices, diags_data, level, BSGS_RATIO, "none")

        ctxt = encrypt_values(scheme, values, level=level)
        ctxt_out_id = scheme.backend.EvaluateLinearTransform(
            lt_id, ctxt.ids[0])
        ctxt_out_id = scheme.backend.Rescale(ctxt_out_id)

        ptxt_out_id = scheme.backend.Decrypt(ctxt_out_id)
        result = scheme.backend.Decode(ptxt_out_id)

        # Compute expected cleartext result
        expected = []
        for i in range(slots):
            val = sum(w * values[(i + d) % slots]
                      for d, w in zip(diag_indices, weights))
            expected.append(val)

        assert_fhe_close(result, expected, atol=1.0,
                         msg=f"BSGS stress {num_diags} diagonals")
        scheme.backend.DeleteLinearTransform(lt_id)

    def test_bsgs_vs_naive_consistency(self, scheme, slots, max_level):
        """Verify BSGS and naive produce the same results."""
        import numpy as np
        rng = np.random.default_rng(seed=99)

        values = [float(i % 13) * 0.2 for i in range(slots)]
        level = max_level

        num_diags = 30
        diag_indices = sorted(rng.choice(slots, size=num_diags,
                                         replace=False).tolist())
        weights = (rng.uniform(-0.3, 0.3, size=num_diags)).tolist()

        diags_data = []
        for w in weights:
            diags_data.extend([w] * slots)

        # Evaluate with BSGS (bsgs_ratio=2)
        lt_bsgs = scheme.backend.GenerateLinearTransform(
            diag_indices, diags_data, level, BSGS_RATIO, "none")
        ctxt1 = encrypt_values(scheme, values, level=level)
        ct_bsgs = scheme.backend.EvaluateLinearTransform(
            lt_bsgs, ctxt1.ids[0])
        ct_bsgs = scheme.backend.Rescale(ct_bsgs)
        pt_bsgs = scheme.backend.Decrypt(ct_bsgs)
        result_bsgs = scheme.backend.Decode(pt_bsgs)

        # Evaluate with naive (bsgs_ratio="none")
        lt_naive = scheme.backend.GenerateLinearTransform(
            diag_indices, diags_data, level, "none", "none")
        ctxt2 = encrypt_values(scheme, values, level=level)
        ct_naive = scheme.backend.EvaluateLinearTransform(
            lt_naive, ctxt2.ids[0])
        ct_naive = scheme.backend.Rescale(ct_naive)
        pt_naive = scheme.backend.Decrypt(ct_naive)
        result_naive = scheme.backend.Decode(pt_naive)

        # Both should be close to each other
        assert_fhe_close(result_bsgs, result_naive, atol=1.0,
                         msg="BSGS vs naive consistency")

        scheme.backend.DeleteLinearTransform(lt_bsgs)
        scheme.backend.DeleteLinearTransform(lt_naive)
