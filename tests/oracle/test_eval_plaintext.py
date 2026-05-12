"""Tests for plaintext-ciphertext arithmetic (add, sub, mul)."""

import torch
import pytest
from fhe_test_utils import assert_fhe_close, encrypt_values, decrypt_values


class TestAddPlaintext:

    def test_add_plaintext(self, scheme, max_level):
        a_vals = [1.0, 2.0, 3.0, 4.0]
        b_vals = [10.0, 20.0, 30.0, 40.0]

        ctxt_a = encrypt_values(scheme, a_vals)
        ptxt_b = scheme.encode(b_vals)

        ctxt_out = ctxt_a + ptxt_b
        result = decrypt_values(scheme, ctxt_out)
        expected = [a + b for a, b in zip(a_vals, b_vals)]
        assert_fhe_close(result, expected, atol=1e-2, msg="add plaintext")

    def test_add_plaintext_negative(self, scheme):
        a_vals = [5.0, 10.0, 15.0, 20.0]
        b_vals = [-5.0, -10.0, -15.0, -20.0]

        ctxt_a = encrypt_values(scheme, a_vals)
        ptxt_b = scheme.encode(b_vals)

        ctxt_out = ctxt_a + ptxt_b
        result = decrypt_values(scheme, ctxt_out)
        expected = [0.0, 0.0, 0.0, 0.0]
        assert_fhe_close(result, expected, atol=1e-2, msg="add plaintext cancel")

    def test_add_plaintext_in_place(self, scheme):
        a_vals = [1.0, 2.0, 3.0, 4.0]
        b_vals = [10.0, 20.0, 30.0, 40.0]

        ctxt_a = encrypt_values(scheme, a_vals)
        ptxt_b = scheme.encode(b_vals)

        ctxt_a += ptxt_b
        result = decrypt_values(scheme, ctxt_a)
        expected = [a + b for a, b in zip(a_vals, b_vals)]
        assert_fhe_close(result, expected, atol=1e-2, msg="iadd plaintext")


class TestSubPlaintext:

    def test_sub_plaintext(self, scheme):
        a_vals = [10.0, 20.0, 30.0, 40.0]
        b_vals = [1.0, 2.0, 3.0, 4.0]

        ctxt_a = encrypt_values(scheme, a_vals)
        ptxt_b = scheme.encode(b_vals)

        ctxt_out = ctxt_a - ptxt_b
        result = decrypt_values(scheme, ctxt_out)
        expected = [a - b for a, b in zip(a_vals, b_vals)]
        assert_fhe_close(result, expected, atol=1e-2, msg="sub plaintext")

    def test_sub_plaintext_in_place(self, scheme):
        a_vals = [10.0, 20.0, 30.0, 40.0]
        b_vals = [1.0, 2.0, 3.0, 4.0]

        ctxt_a = encrypt_values(scheme, a_vals)
        ptxt_b = scheme.encode(b_vals)

        ctxt_a -= ptxt_b
        result = decrypt_values(scheme, ctxt_a)
        expected = [a - b for a, b in zip(a_vals, b_vals)]
        assert_fhe_close(result, expected, atol=1e-2, msg="isub plaintext")


class TestMulPlaintext:

    def test_mul_plaintext(self, scheme):
        """Ciphertext * plaintext, consumes one level (rescale)."""
        a_vals = [1.0, 2.0, 3.0, 4.0]
        b_vals = [2.0, 3.0, 4.0, 5.0]

        ctxt_a = encrypt_values(scheme, a_vals)
        ptxt_b = scheme.encode(b_vals)

        ctxt_out = ctxt_a * ptxt_b
        result = decrypt_values(scheme, ctxt_out)
        expected = [a * b for a, b in zip(a_vals, b_vals)]
        assert_fhe_close(result, expected, atol=1e-1, msg="mul plaintext")

    def test_mul_plaintext_consumes_level(self, scheme, max_level):
        """MulPlaintext + rescale should drop the ciphertext level by 1."""
        a_vals = [1.0, 2.0, 3.0, 4.0]
        b_vals = [2.0, 3.0, 4.0, 5.0]

        ctxt_a = encrypt_values(scheme, a_vals)
        level_before = ctxt_a.level()

        ptxt_b = scheme.encode(b_vals)
        ctxt_out = ctxt_a * ptxt_b

        assert ctxt_out.level() == level_before - 1, (
            f"Expected level {level_before - 1} after mul+rescale, "
            f"got {ctxt_out.level()}")

    def test_mul_plaintext_in_place(self, scheme):
        a_vals = [1.0, 2.0, 3.0, 4.0]
        b_vals = [2.0, 3.0, 4.0, 5.0]

        ctxt_a = encrypt_values(scheme, a_vals)
        ptxt_b = scheme.encode(b_vals)

        ctxt_a *= ptxt_b
        result = decrypt_values(scheme, ctxt_a)
        expected = [a * b for a, b in zip(a_vals, b_vals)]
        assert_fhe_close(result, expected, atol=1e-1, msg="imul plaintext")
