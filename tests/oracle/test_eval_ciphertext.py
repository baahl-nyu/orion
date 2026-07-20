"""Tests for ciphertext-ciphertext arithmetic (add, sub, mul)."""

import torch
import pytest
from fhe_test_utils import assert_fhe_close, encrypt_values, decrypt_values


class TestAddCiphertext:

    def test_add_ciphertext(self, scheme):
        a_vals = [1.0, 2.0, 3.0, 4.0]
        b_vals = [10.0, 20.0, 30.0, 40.0]

        ctxt_a = encrypt_values(scheme, a_vals)
        ctxt_b = encrypt_values(scheme, b_vals)

        ctxt_out = ctxt_a + ctxt_b
        result = decrypt_values(scheme, ctxt_out)
        expected = [a + b for a, b in zip(a_vals, b_vals)]
        assert_fhe_close(result, expected, atol=1e-2, msg="add ciphertext")

    def test_add_ciphertext_self(self, scheme):
        """ct + ct should equal 2*ct."""
        values = [1.0, 2.0, 3.0, 4.0]
        ctxt = encrypt_values(scheme, values)

        ctxt_out = ctxt + ctxt
        result = decrypt_values(scheme, ctxt_out)
        expected = [2 * v for v in values]
        assert_fhe_close(result, expected, atol=1e-2, msg="add self")

    def test_add_ciphertext_in_place(self, scheme):
        a_vals = [1.0, 2.0, 3.0, 4.0]
        b_vals = [10.0, 20.0, 30.0, 40.0]

        ctxt_a = encrypt_values(scheme, a_vals)
        ctxt_b = encrypt_values(scheme, b_vals)

        ctxt_a += ctxt_b
        result = decrypt_values(scheme, ctxt_a)
        expected = [a + b for a, b in zip(a_vals, b_vals)]
        assert_fhe_close(result, expected, atol=1e-2, msg="iadd ciphertext")


class TestSubCiphertext:

    def test_sub_ciphertext(self, scheme):
        a_vals = [10.0, 20.0, 30.0, 40.0]
        b_vals = [1.0, 2.0, 3.0, 4.0]

        ctxt_a = encrypt_values(scheme, a_vals)
        ctxt_b = encrypt_values(scheme, b_vals)

        ctxt_out = ctxt_a - ctxt_b
        result = decrypt_values(scheme, ctxt_out)
        expected = [a - b for a, b in zip(a_vals, b_vals)]
        assert_fhe_close(result, expected, atol=1e-2, msg="sub ciphertext")

    def test_sub_ciphertext_self(self, scheme):
        """ct - ct should be ~0."""
        values = [1.0, 2.0, 3.0, 4.0]
        ctxt = encrypt_values(scheme, values)

        ctxt_out = ctxt - ctxt
        result = decrypt_values(scheme, ctxt_out)
        expected = [0.0] * len(values)
        assert_fhe_close(result, expected, atol=1e-2, msg="sub self")

    def test_sub_ciphertext_in_place(self, scheme):
        a_vals = [10.0, 20.0, 30.0, 40.0]
        b_vals = [1.0, 2.0, 3.0, 4.0]

        ctxt_a = encrypt_values(scheme, a_vals)
        ctxt_b = encrypt_values(scheme, b_vals)

        ctxt_a -= ctxt_b
        result = decrypt_values(scheme, ctxt_a)
        expected = [a - b for a, b in zip(a_vals, b_vals)]
        assert_fhe_close(result, expected, atol=1e-2, msg="isub ciphertext")


class TestMulCiphertext:

    def test_mul_ciphertext(self, scheme):
        """Ciphertext * ciphertext (with relin + rescale)."""
        a_vals = [1.0, 2.0, 3.0, 4.0]
        b_vals = [2.0, 3.0, 4.0, 5.0]

        ctxt_a = encrypt_values(scheme, a_vals)
        ctxt_b = encrypt_values(scheme, b_vals)

        ctxt_out = ctxt_a * ctxt_b
        result = decrypt_values(scheme, ctxt_out)
        expected = [a * b for a, b in zip(a_vals, b_vals)]
        assert_fhe_close(result, expected, atol=1e-1, msg="mul ciphertext")

    def test_mul_ciphertext_consumes_level(self, scheme, max_level):
        """Mul + relin + rescale should drop level by 1."""
        a_vals = [1.0, 2.0, 3.0, 4.0]
        b_vals = [2.0, 3.0, 4.0, 5.0]

        ctxt_a = encrypt_values(scheme, a_vals)
        ctxt_b = encrypt_values(scheme, b_vals)
        level_before = ctxt_a.level()

        ctxt_out = ctxt_a * ctxt_b
        assert ctxt_out.level() == level_before - 1, (
            f"Expected level {level_before - 1}, got {ctxt_out.level()}")

    def test_mul_ciphertext_square(self, scheme):
        """ct * ct (squaring)."""
        values = [1.0, 2.0, 3.0, 4.0]
        ctxt = encrypt_values(scheme, values)

        ctxt_out = ctxt * ctxt
        result = decrypt_values(scheme, ctxt_out)
        expected = [v * v for v in values]
        assert_fhe_close(result, expected, atol=1e-1, msg="square")

    def test_mul_ciphertext_in_place(self, scheme):
        a_vals = [1.0, 2.0, 3.0, 4.0]
        b_vals = [2.0, 3.0, 4.0, 5.0]

        ctxt_a = encrypt_values(scheme, a_vals)
        ctxt_b = encrypt_values(scheme, b_vals)

        ctxt_a *= ctxt_b
        result = decrypt_values(scheme, ctxt_a)
        expected = [a * b for a, b in zip(a_vals, b_vals)]
        assert_fhe_close(result, expected, atol=1e-1, msg="imul ciphertext")
