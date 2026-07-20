"""Tests for polynomial evaluation (monomial and Chebyshev basis)."""

import math
import torch
import numpy as np
import pytest
from fhe_test_utils import assert_fhe_close, encrypt_values, decrypt_values


class TestMonomialPolynomial:

    def test_monomial_linear(self, scheme, max_level):
        """Evaluate f(x) = 2x + 1 (degree-1 monomial).

        generate_monomial takes descending order: [a_n, ..., a_1, a_0].
        So [2, 1] means f(x) = 2x + 1.
        """
        coeffs = [2.0, 1.0]  # descending: a1=2, a0=1 -> f(x) = 2x + 1
        poly_id = scheme.poly_evaluator.generate_monomial(coeffs)

        values = [0.5, 1.0, -0.5, 0.0]
        ctxt = encrypt_values(scheme, values)
        ctxt_out = scheme.poly_evaluator.evaluate_polynomial(ctxt, poly_id)
        result = decrypt_values(scheme, ctxt_out)

        expected = [2.0 * v + 1.0 for v in values]
        assert_fhe_close(result, expected, atol=1e-1, msg="monomial linear")

    def test_monomial_quadratic(self, scheme, max_level):
        """Evaluate f(x) = x^2 + x + 1.

        Descending: [a2, a1, a0] = [1, 1, 1].
        """
        coeffs = [1.0, 1.0, 1.0]  # descending: a2=1, a1=1, a0=1
        poly_id = scheme.poly_evaluator.generate_monomial(coeffs)

        values = [0.5, 1.0, -0.5, 0.0]
        ctxt = encrypt_values(scheme, values)
        ctxt_out = scheme.poly_evaluator.evaluate_polynomial(ctxt, poly_id)
        result = decrypt_values(scheme, ctxt_out)

        expected = [v**2 + v + 1.0 for v in values]
        assert_fhe_close(result, expected, atol=1e-1, msg="monomial quadratic")

    def test_monomial_cubic(self, scheme, max_level):
        """Evaluate f(x) = x^3 - x (odd function).

        Descending: [a3, a2, a1, a0] = [1, 0, -1, 0].
        """
        coeffs = [1.0, 0.0, -1.0, 0.0]  # descending: a3=1, a2=0, a1=-1, a0=0
        poly_id = scheme.poly_evaluator.generate_monomial(coeffs)

        values = [0.25, 0.5, -0.25, -0.5]
        ctxt = encrypt_values(scheme, values)
        ctxt_out = scheme.poly_evaluator.evaluate_polynomial(ctxt, poly_id)
        result = decrypt_values(scheme, ctxt_out)

        expected = [v**3 - v for v in values]
        assert_fhe_close(result, expected, atol=1e-1, msg="monomial cubic")


class TestChebyshevPolynomial:

    def test_chebyshev_degree2(self, scheme, max_level):
        """Evaluate a degree-2 Chebyshev polynomial.

        T0(x) = 1, T1(x) = x, T2(x) = 2x^2 - 1
        Coeffs [c0, c1, c2] means f(x) = c0*T0 + c1*T1 + c2*T2
        With [1, 0, 1]: f(x) = 1 + (2x^2 - 1) = 2x^2
        """
        coeffs = [1.0, 0.0, 1.0]
        poly_id = scheme.poly_evaluator.generate_chebyshev(coeffs)

        values = [0.25, 0.5, -0.25, -0.5]
        ctxt = encrypt_values(scheme, values)
        ctxt_out = scheme.poly_evaluator.evaluate_polynomial(ctxt, poly_id)
        result = decrypt_values(scheme, ctxt_out)

        expected = [2.0 * v**2 for v in values]
        assert_fhe_close(result, expected, atol=1e-1, msg="chebyshev deg-2")

    def test_chebyshev_identity(self, scheme, max_level):
        """T1(x) = x, so coeffs [0, 1] should give back x."""
        coeffs = [0.0, 1.0]
        poly_id = scheme.poly_evaluator.generate_chebyshev(coeffs)

        values = [0.1, 0.5, -0.3, 0.9]
        ctxt = encrypt_values(scheme, values)
        ctxt_out = scheme.poly_evaluator.evaluate_polynomial(ctxt, poly_id)
        result = decrypt_values(scheme, ctxt_out)

        assert_fhe_close(result, values, atol=1e-1, msg="chebyshev identity")


class TestMinimaxSign:

    def test_minimax_sign_returns_coeffs(self, scheme):
        """generate_minimax_sign_coeffs should return coefficient tensors."""
        degrees = [7]
        coeffs_list = scheme.poly_evaluator.generate_minimax_sign_coeffs(
            degrees, prec=64, logalpha=6, logerr=6)

        assert len(coeffs_list) == 1, (
            f"Expected 1 set of coefficients, got {len(coeffs_list)}")
        assert len(coeffs_list[0]) == 8, (
            f"Degree-7 poly should have 8 coefficients, got {len(coeffs_list[0])}")

    def test_minimax_sign_multi_degree(self, scheme):
        """Multiple degrees should return one tensor per degree."""
        degrees = [3, 5]
        coeffs_list = scheme.poly_evaluator.generate_minimax_sign_coeffs(
            degrees, prec=64, logalpha=6, logerr=6)

        assert len(coeffs_list) == 2
        assert len(coeffs_list[0]) == 4  # degree 3 -> 4 coeffs
        assert len(coeffs_list[1]) == 6  # degree 5 -> 6 coeffs
