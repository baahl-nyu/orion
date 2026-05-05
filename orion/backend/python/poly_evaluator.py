import torch 
import numpy as np

from .tensors import CipherTensor

class NewEvaluator:
    def __init__(self, scheme):
        self.scheme = scheme 
        self.backend = scheme.backend
        self.new_polynomial_evaluator()

    def new_polynomial_evaluator(self):
        self.backend.NewPolynomialEvaluator()

    def generate_monomial(self, coeffs):
        if isinstance(coeffs, (torch.Tensor, np.ndarray)):
            coeffs = coeffs.tolist()
        return self.backend.GenerateMonomial(coeffs[::-1])
    
    def generate_chebyshev(self, coeffs):
        if isinstance(coeffs, (torch.Tensor, np.ndarray)):
            coeffs = coeffs.tolist()
        return self.backend.GenerateChebyshev(coeffs)

    def evaluate_polynomial(self, ciphertensor, poly, out_scale=None):
        out_scale = out_scale or self.scheme.params.get_default_scale()

        cts_out = []  
        for ctxt in ciphertensor.ids:
            ct_out = self.backend.EvaluatePolynomial(ctxt, poly, out_scale)
            cts_out.append(ct_out)

        return CipherTensor(
            self.scheme, cts_out, ciphertensor.shape, ciphertensor.on_shape, ciphertensor.start, ciphertensor.stride, ciphertensor.stop)
    
    def generate_minimax_sign_coeffs(self, degrees, prec=128, logalpha=12, 
                                     logerr=12, debug=False):
        if isinstance(degrees, int):
            degrees = [degrees]
        else:
            degrees = list(degrees)

        degrees = [d for d in degrees if d != 0]
        if len(degrees) == 0:
            raise ValueError(
                "At least one non-zero degree polynomial must be provided to "
                "generate_minimax_sign_coeffs(). "
            )

        coeffs_flat = self.backend.GenerateMinimaxSignCoeffs(
            degrees, prec, logalpha, logerr, int(debug)
        )

        coeffs_flat = torch.tensor(coeffs_flat)
        splits = [degree + 1 for degree in degrees]
        return torch.split(coeffs_flat, splits)

    def get_depth(self, poly):
        return self.backend.GetPolyDepth(poly)
