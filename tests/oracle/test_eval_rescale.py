"""Tests for rescale and scale/level metadata management."""

import torch
import pytest
from fhe_test_utils import assert_fhe_close, encrypt_values, decrypt_values


class TestRescale:

    def test_rescale_drops_level(self, scheme, max_level):
        """Manual rescale should drop the ciphertext level by 1."""
        values = [1.0, 2.0, 3.0, 4.0]
        ctxt = encrypt_values(scheme, values)
        level_before = ctxt.level()

        # MulScalarFloat triggers an automatic rescale, but we can also
        # test it via the evaluator directly after a plaintext multiply
        # that does NOT auto-rescale. We'll use the backend directly.
        ctxt_id = scheme.backend.MulScalarFloatNew(ctxt.ids[0], 2.0)
        # Now manually rescale
        ctxt_id = scheme.backend.Rescale(ctxt_id)

        level_after = scheme.backend.GetCiphertextLevel(ctxt_id)
        assert level_after == level_before - 1, (
            f"Expected level {level_before - 1} after rescale, got {level_after}")


class TestScaleManagement:

    def test_get_ciphertext_scale(self, scheme, default_scale):
        """Freshly encrypted ciphertext should have the default scale."""
        values = [1.0, 2.0]
        ctxt = encrypt_values(scheme, values)
        scale = ctxt.scale()
        assert scale == default_scale, (
            f"Expected scale {default_scale}, got {scale}")

    def test_set_ciphertext_scale(self, scheme, default_scale):
        """Setting scale should be reflected when queried."""
        values = [1.0, 2.0]
        ctxt = encrypt_values(scheme, values)
        new_scale = default_scale * 2
        ctxt.set_scale(new_scale)
        assert ctxt.scale() == new_scale, (
            f"Expected scale {new_scale}, got {ctxt.scale()}")

    def test_get_plaintext_scale(self, scheme, default_scale):
        """Freshly encoded plaintext should have the default scale."""
        values = [1.0, 2.0]
        ptxt = scheme.encode(values)
        scale = ptxt.scale()
        assert scale == default_scale, (
            f"Expected scale {default_scale}, got {scale}")

    def test_set_plaintext_scale(self, scheme, default_scale):
        """Setting plaintext scale should be reflected when queried."""
        values = [1.0, 2.0]
        ptxt = scheme.encode(values)
        new_scale = default_scale * 2
        ptxt.set_scale(new_scale)
        assert ptxt.scale() == new_scale, (
            f"Expected scale {new_scale}, got {ptxt.scale()}")


class TestLevelManagement:

    def test_level_decreases_after_mul_float_scalar(self, scheme, max_level):
        """MulScalarFloat (which auto-rescales) should drop level by 1."""
        values = [1.0, 2.0, 3.0, 4.0]
        ctxt = encrypt_values(scheme, values)
        level_before = ctxt.level()
        ctxt_out = ctxt * 2.5  # float triggers rescale
        assert ctxt_out.level() == level_before - 1

    def test_level_stable_after_add(self, scheme, max_level):
        """Addition should not change the ciphertext level."""
        values = [1.0, 2.0, 3.0, 4.0]
        ctxt = encrypt_values(scheme, values)
        level_before = ctxt.level()
        ctxt_out = ctxt + 5.0
        assert ctxt_out.level() == level_before

    def test_level_stable_after_int_mul(self, scheme, max_level):
        """MulScalarInt should not change the ciphertext level."""
        values = [1.0, 2.0, 3.0, 4.0]
        ctxt = encrypt_values(scheme, values)
        level_before = ctxt.level()
        ctxt_out = ctxt * 3  # int, no rescale
        assert ctxt_out.level() == level_before

    def test_successive_muls_drain_levels(self, scheme, max_level):
        """Chained float multiplies should drain levels one at a time."""
        values = [1.0, 2.0, 3.0, 4.0]
        ctxt = encrypt_values(scheme, values)

        for i in range(max_level - 1):
            ctxt = ctxt * 1.5  # each costs 1 level
            expected_level = max_level - (i + 1)
            assert ctxt.level() == expected_level, (
                f"After {i+1} muls: expected level {expected_level}, "
                f"got {ctxt.level()}")
