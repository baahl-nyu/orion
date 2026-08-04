"""Tests for linear transform (diagonal matrix-vector multiply)."""

import torch
import pytest
from fhe_test_utils import assert_fhe_close, encrypt_values, decrypt_values


class TestLinearTransform:

    def test_identity_transform(self, scheme, slots, max_level):
        """A diagonal matrix with all 1s on the main diagonal = identity."""
        values = [float(i + 1) for i in range(slots)]
        level = max_level

        # Identity: single diagonal at index 0, all ones
        diags_idxs = [0]
        diags_data = [1.0] * slots

        lt_id = scheme.backend.GenerateLinearTransform(
            diags_idxs, diags_data, level, 1.0, "none")

        # Generate rotation keys for this transform
        rot_keys = scheme.backend.GetLinearTransformRotationKeys(lt_id)
        for key in rot_keys:
            scheme.backend.GenerateLinearTransformRotationKey(key)

        ctxt = encrypt_values(scheme, values, level=level)
        ctxt_out_id = scheme.backend.EvaluateLinearTransform(lt_id, ctxt.ids[0])

        # Rescale after linear transform
        # (backends whose LT output is already rescaled -- cheddar's
        # double-hoisted path -- advertise it via lt_output_prerescaled)
        if not getattr(scheme.backend, "lt_output_prerescaled", False):
            ctxt_out_id = scheme.backend.Rescale(ctxt_out_id)

        # Decrypt the raw ciphertext ID
        ptxt_out_id = scheme.backend.Decrypt(ctxt_out_id)
        result = scheme.backend.Decode(ptxt_out_id)

        assert_fhe_close(result, values, atol=1e-1, msg="identity transform")
        scheme.backend.DeleteLinearTransform(lt_id)

    def test_scale_by_constant(self, scheme, slots, max_level):
        """Diagonal matrix with constant c on the main diagonal = c * x."""
        values = [float(i + 1) for i in range(slots)]
        c = 3.0
        level = max_level

        diags_idxs = [0]
        diags_data = [c] * slots

        lt_id = scheme.backend.GenerateLinearTransform(
            diags_idxs, diags_data, level, 1.0, "none")

        rot_keys = scheme.backend.GetLinearTransformRotationKeys(lt_id)
        for key in rot_keys:
            scheme.backend.GenerateLinearTransformRotationKey(key)

        ctxt = encrypt_values(scheme, values, level=level)
        ctxt_out_id = scheme.backend.EvaluateLinearTransform(lt_id, ctxt.ids[0])
        # (backends whose LT output is already rescaled -- cheddar's
        # double-hoisted path -- advertise it via lt_output_prerescaled)
        if not getattr(scheme.backend, "lt_output_prerescaled", False):
            ctxt_out_id = scheme.backend.Rescale(ctxt_out_id)

        ptxt_out_id = scheme.backend.Decrypt(ctxt_out_id)
        result = scheme.backend.Decode(ptxt_out_id)

        expected = [c * v for v in values]
        assert_fhe_close(result, expected, atol=5e-1, msg="scale transform")
        scheme.backend.DeleteLinearTransform(lt_id)

    def test_cyclic_permutation(self, scheme, slots, max_level):
        """A single diagonal at index k = cyclic shift by k positions."""
        values = [float(i + 1) for i in range(slots)]
        k = 3
        level = max_level

        diags_idxs = [k]
        diags_data = [1.0] * slots

        lt_id = scheme.backend.GenerateLinearTransform(
            diags_idxs, diags_data, level, 1.0, "none")

        rot_keys = scheme.backend.GetLinearTransformRotationKeys(lt_id)
        for key in rot_keys:
            scheme.backend.GenerateLinearTransformRotationKey(key)

        ctxt = encrypt_values(scheme, values, level=level)
        ctxt_out_id = scheme.backend.EvaluateLinearTransform(lt_id, ctxt.ids[0])
        # (backends whose LT output is already rescaled -- cheddar's
        # double-hoisted path -- advertise it via lt_output_prerescaled)
        if not getattr(scheme.backend, "lt_output_prerescaled", False):
            ctxt_out_id = scheme.backend.Rescale(ctxt_out_id)

        ptxt_out_id = scheme.backend.Decrypt(ctxt_out_id)
        result = scheme.backend.Decode(ptxt_out_id)

        # Diagonal at index k means output[i] = diag[i] * input[(i+k) mod n]
        expected = [values[(i + k) % slots] for i in range(slots)]
        assert_fhe_close(result, expected, atol=1e-1, msg=f"permute by {k}")
        scheme.backend.DeleteLinearTransform(lt_id)

    def test_multi_diagonal(self, scheme, slots, max_level):
        """Sum of two diagonals: identity + shift-by-1."""
        values = [float(i + 1) for i in range(slots)]
        level = max_level

        # Diagonal 0 (identity) + diagonal 1 (shift by 1)
        diags_idxs = [0, 1]
        diags_data = [1.0] * slots + [1.0] * slots  # flattened

        lt_id = scheme.backend.GenerateLinearTransform(
            diags_idxs, diags_data, level, 1.0, "none")

        rot_keys = scheme.backend.GetLinearTransformRotationKeys(lt_id)
        for key in rot_keys:
            scheme.backend.GenerateLinearTransformRotationKey(key)

        ctxt = encrypt_values(scheme, values, level=level)
        ctxt_out_id = scheme.backend.EvaluateLinearTransform(lt_id, ctxt.ids[0])
        # (backends whose LT output is already rescaled -- cheddar's
        # double-hoisted path -- advertise it via lt_output_prerescaled)
        if not getattr(scheme.backend, "lt_output_prerescaled", False):
            ctxt_out_id = scheme.backend.Rescale(ctxt_out_id)

        ptxt_out_id = scheme.backend.Decrypt(ctxt_out_id)
        result = scheme.backend.Decode(ptxt_out_id)

        # output[i] = input[i] + input[(i+1) mod n]
        expected = [values[i] + values[(i + 1) % slots] for i in range(slots)]
        assert_fhe_close(result, expected, atol=5e-1, msg="multi-diagonal")
        scheme.backend.DeleteLinearTransform(lt_id)

    def test_input_above_transform_level(self, scheme, slots, max_level):
        """Input ciphertext arrives ABOVE the transform's level.

        This is the normal mid-network case (a ciphertext descends into a
        transform whose plaintexts were encoded lower) and exercises the
        backend's input alignment: the cheddar backend drops the input to
        the transform level once, before its rotations, instead of
        re-aligning every rotated operand per accumulation
        (bindings.EvaluateLinearTransform -> _align_lt_input).
        """
        if max_level < 2:
            pytest.skip("needs at least one level of headroom")
        values = [float(i % 7) - 3.0 for i in range(slots)]
        lt_level = 1  # transform low, input at the top of the chain

        diags_idxs = [0, 1]
        diags_data = [2.0] * slots + [1.0] * slots
        lt_id = scheme.backend.GenerateLinearTransform(
            diags_idxs, diags_data, lt_level, 1.0, "none")

        for key in scheme.backend.GetLinearTransformRotationKeys(lt_id):
            scheme.backend.GenerateLinearTransformRotationKey(key)

        ctxt = encrypt_values(scheme, values, level=max_level)
        ctxt_out_id = scheme.backend.EvaluateLinearTransform(lt_id, ctxt.ids[0])
        if not getattr(scheme.backend, "lt_output_prerescaled", False):
            ctxt_out_id = scheme.backend.Rescale(ctxt_out_id)

        result = scheme.backend.Decode(scheme.backend.Decrypt(ctxt_out_id))
        expected = [2.0 * values[i] + values[(i + 1) % slots]
                    for i in range(slots)]
        assert_fhe_close(result, expected, atol=5e-1,
                         msg="input above transform level")
        scheme.backend.DeleteLinearTransform(lt_id)

    def test_input_above_transform_level_bsgs(self, scheme, slots, max_level):
        """Same gap case through the BSGS path (bsgs_ratio=2, several
        diagonals -> baby-step batch + giant-step rotations all run at
        the transform level after the single input alignment)."""
        if max_level < 2:
            pytest.skip("needs at least one level of headroom")
        values = [float((i * 5) % 11) - 5.0 for i in range(slots)]
        lt_level = 1

        diags_idxs = [0, 1, 2, 5, 8]
        diags_data = []
        for d in diags_idxs:
            diags_data += [float(d + 1)] * slots
        lt_id = scheme.backend.GenerateLinearTransform(
            diags_idxs, diags_data, lt_level, 2.0, "none")

        for key in scheme.backend.GetLinearTransformRotationKeys(lt_id):
            scheme.backend.GenerateLinearTransformRotationKey(key)

        ctxt = encrypt_values(scheme, values, level=max_level)
        ctxt_out_id = scheme.backend.EvaluateLinearTransform(lt_id, ctxt.ids[0])
        if not getattr(scheme.backend, "lt_output_prerescaled", False):
            ctxt_out_id = scheme.backend.Rescale(ctxt_out_id)

        result = scheme.backend.Decode(scheme.backend.Decrypt(ctxt_out_id))
        expected = [sum((d + 1.0) * values[(i + d) % slots]
                        for d in diags_idxs) for i in range(slots)]
        assert_fhe_close(result, expected, atol=5e-1,
                         msg="input above transform level (bsgs)")
        scheme.backend.DeleteLinearTransform(lt_id)
