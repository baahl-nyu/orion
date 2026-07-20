"""Tests for scalar arithmetic on ciphertexts (add, sub, mul)."""

import torch
import pytest
from fhe_test_utils import assert_fhe_close, encrypt_values, decrypt_values


class TestAddScalar:

    def test_add_scalar_positive(self, scheme):
        values = [1.0, 2.0, 3.0, 4.0]
        ctxt = encrypt_values(scheme, values)
        ctxt_out = ctxt + 5.0
        result = decrypt_values(scheme, ctxt_out)
        expected = [v + 5.0 for v in values]
        assert_fhe_close(result, expected, atol=1e-2, msg="add scalar +5")

    def test_add_scalar_negative(self, scheme):
        values = [1.0, 2.0, 3.0, 4.0]
        ctxt = encrypt_values(scheme, values)
        ctxt_out = ctxt + (-3.0)
        result = decrypt_values(scheme, ctxt_out)
        expected = [v - 3.0 for v in values]
        assert_fhe_close(result, expected, atol=1e-2, msg="add scalar -3")

    def test_add_scalar_zero(self, scheme):
        values = [1.0, 2.0, 3.0, 4.0]
        ctxt = encrypt_values(scheme, values)
        ctxt_out = ctxt + 0.0
        result = decrypt_values(scheme, ctxt_out)
        assert_fhe_close(result, values, atol=1e-2, msg="add scalar 0")

    def test_add_scalar_in_place(self, scheme):
        values = [1.0, 2.0, 3.0, 4.0]
        ctxt = encrypt_values(scheme, values)
        ctxt += 5.0
        result = decrypt_values(scheme, ctxt)
        expected = [v + 5.0 for v in values]
        assert_fhe_close(result, expected, atol=1e-2, msg="iadd scalar")


class TestSubScalar:

    def test_sub_scalar_positive(self, scheme):
        values = [10.0, 20.0, 30.0, 40.0]
        ctxt = encrypt_values(scheme, values)
        ctxt_out = ctxt - 5.0
        result = decrypt_values(scheme, ctxt_out)
        expected = [v - 5.0 for v in values]
        assert_fhe_close(result, expected, atol=1e-2, msg="sub scalar 5")

    def test_sub_scalar_negative(self, scheme):
        values = [1.0, 2.0, 3.0, 4.0]
        ctxt = encrypt_values(scheme, values)
        ctxt_out = ctxt - (-2.0)
        result = decrypt_values(scheme, ctxt_out)
        expected = [v + 2.0 for v in values]
        assert_fhe_close(result, expected, atol=1e-2, msg="sub scalar -2")

    def test_sub_scalar_in_place(self, scheme):
        values = [10.0, 20.0, 30.0, 40.0]
        ctxt = encrypt_values(scheme, values)
        ctxt -= 5.0
        result = decrypt_values(scheme, ctxt)
        expected = [v - 5.0 for v in values]
        assert_fhe_close(result, expected, atol=1e-2, msg="isub scalar")


class TestMulScalar:

    def test_mul_scalar_float(self, scheme):
        values = [1.0, 2.0, 3.0, 4.0]
        ctxt = encrypt_values(scheme, values)
        ctxt_out = ctxt * 2.5
        result = decrypt_values(scheme, ctxt_out)
        expected = [v * 2.5 for v in values]
        assert_fhe_close(result, expected, atol=1e-1, msg="mul scalar float 2.5")

    def test_mul_scalar_int(self, scheme):
        values = [1.0, 2.0, 3.0, 4.0]
        ctxt = encrypt_values(scheme, values)
        ctxt_out = ctxt * 3
        result = decrypt_values(scheme, ctxt_out)
        expected = [v * 3 for v in values]
        assert_fhe_close(result, expected, atol=1e-2, msg="mul scalar int 3")

    def test_mul_scalar_negative(self, scheme):
        values = [1.0, 2.0, 3.0, 4.0]
        ctxt = encrypt_values(scheme, values)
        ctxt_out = ctxt * (-2.0)
        result = decrypt_values(scheme, ctxt_out)
        expected = [v * -2.0 for v in values]
        assert_fhe_close(result, expected, atol=1e-1, msg="mul scalar -2")

    def test_mul_scalar_one(self, scheme):
        """Multiplying by 1 (int) should be a no-op."""
        values = [1.0, 2.0, 3.0, 4.0]
        ctxt = encrypt_values(scheme, values)
        ctxt_out = ctxt * 1
        result = decrypt_values(scheme, ctxt_out)
        assert_fhe_close(result, values, atol=1e-2, msg="mul scalar 1")

    def test_mul_scalar_in_place(self, scheme):
        values = [1.0, 2.0, 3.0, 4.0]
        ctxt = encrypt_values(scheme, values)
        ctxt *= 2.5
        result = decrypt_values(scheme, ctxt)
        expected = [v * 2.5 for v in values]
        assert_fhe_close(result, expected, atol=1e-1, msg="imul scalar")
