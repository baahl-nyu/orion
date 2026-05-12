"""Tests for rotation and negation on ciphertexts."""

import torch
import pytest
from fhe_test_utils import assert_fhe_close, encrypt_values, decrypt_values


class TestNegate:

    def test_negate(self, scheme):
        values = [1.0, -2.0, 3.0, -4.0]
        ctxt = encrypt_values(scheme, values)
        ctxt_out = -ctxt
        result = decrypt_values(scheme, ctxt_out)
        expected = [-v for v in values]
        assert_fhe_close(result, expected, atol=1e-2, msg="negate")

    def test_double_negate(self, scheme):
        """Negating twice should return original values."""
        values = [1.0, -2.0, 3.0, -4.0]
        ctxt = encrypt_values(scheme, values)
        ctxt_out = -(-ctxt)
        result = decrypt_values(scheme, ctxt_out)
        assert_fhe_close(result, values, atol=1e-2, msg="double negate")


class TestRotate:

    def test_rotate_left_1(self, scheme, slots):
        """Rotate left by 1 (positive k = left-shift in Lattigo CKKS)."""
        values = list(range(1, slots + 1))
        values = [float(v) for v in values]
        ctxt = encrypt_values(scheme, values)

        scheme.evaluator.add_rotation_key(1)
        ctxt_out = ctxt.roll(1)
        result = decrypt_values(scheme, ctxt_out)

        # Cyclic left-shift by 1: first element wraps to end
        expected = values[1:] + [values[0]]
        assert_fhe_close(result, expected, atol=1e-2, msg="rotate left 1")

    def test_rotate_right_1(self, scheme, slots):
        """Rotate right by 1 (negative k = right-shift)."""
        values = list(range(1, slots + 1))
        values = [float(v) for v in values]
        ctxt = encrypt_values(scheme, values)

        scheme.evaluator.add_rotation_key(-1)
        ctxt_out = ctxt.roll(-1)
        result = decrypt_values(scheme, ctxt_out)

        # Cyclic right-shift by 1: last element wraps to front
        expected = [values[-1]] + values[:-1]
        assert_fhe_close(result, expected, atol=1e-2, msg="rotate right 1")

    def test_rotate_by_n(self, scheme, slots):
        """Rotate by n positions (left-shift)."""
        n = 4
        values = list(range(1, slots + 1))
        values = [float(v) for v in values]
        ctxt = encrypt_values(scheme, values)

        scheme.evaluator.add_rotation_key(n)
        ctxt_out = ctxt.roll(n)
        result = decrypt_values(scheme, ctxt_out)

        # Cyclic left-shift by n
        expected = values[n:] + values[:n]
        assert_fhe_close(result, expected, atol=1e-2, msg=f"rotate left {n}")

    def test_rotate_full_cycle(self, scheme, slots):
        """Rotating by the full slot count should return original."""
        values = list(range(1, slots + 1))
        values = [float(v) for v in values]
        ctxt = encrypt_values(scheme, values)

        scheme.evaluator.add_rotation_key(slots)
        ctxt_out = ctxt.roll(slots)
        result = decrypt_values(scheme, ctxt_out)
        assert_fhe_close(result, values, atol=1e-2, msg="rotate full cycle")

    def test_rotate_zero(self, scheme, slots):
        """Rotating by 0 is a no-op."""
        values = [1.0, 2.0, 3.0, 4.0]
        ctxt = encrypt_values(scheme, values)

        scheme.evaluator.add_rotation_key(0)
        ctxt_out = ctxt.roll(0)
        result = decrypt_values(scheme, ctxt_out)
        assert_fhe_close(result, values, atol=1e-2, msg="rotate 0")
