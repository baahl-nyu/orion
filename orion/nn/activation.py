import math 

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from orion.nn.module import Module, timer
from orion.nn.operations import Mult


class Activation(Module):
    def __init__(self, coeffs):
        super().__init__()
        self.coeffs = coeffs 
        self.output_scale = None
        self.set_depth()

    def extra_repr(self):
        return super().extra_repr() + f", degree={len(self.coeffs)-1}"
    
    def set_depth(self):
        self.depth = int(math.ceil(math.log2(len(self.coeffs))))

    def set_output_scale(self, output_scale):
        self.output_scale = output_scale

    def compile(self):
        self.poly = self.scheme.poly_evaluator.generate_monomial(self.coeffs)

    @timer
    def forward(self, x):
        if self.he_mode:
            return self.scheme.poly_evaluator.evaluate_polynomial( 
                x, self.poly, self.output_scale)
        
        # Horner's method
        out = 0
        for coeff in self.coeffs:
            out = coeff + x * out
            
        return out
    

class Quad(Module):
    def __init__(self):
        super().__init__()
        self.set_depth(1)

    def forward(self, x):
        out = x * x 
        if self.he_mode:
            out.set_scale(x.scale()) 
        return out
    

class Chebyshev(Module):
    def __init__(self, degree: int, fn, within_composite=False):
        super().__init__()
        self.degree = degree
        self.fn = fn
        self.within_composite = within_composite
        self.coeffs = None
       
        self.output_scale = None
        self.prescale = 1 
        self.constant = 0

    def extra_repr(self):
        return super().extra_repr() + f", degree={self.degree}"

    def fit(self):
        if not self.within_composite:
            center = (self.input_min + self.input_max) / 2 
            half_range = (self.input_max - self.input_min) / 2
            self.low = (center - (self.margin * half_range)).item()
            self.high = (center + (self.margin * half_range)).item()

            nodes = np.polynomial.chebyshev.chebpts1(self.degree + 1)
            if self.low < -1 or self.high > 1:
                self.prescale = 2 / (self.high - self.low) 
                self.constant = -self.prescale * (self.low + self.high) / 2 
                evals = (nodes + 1) * (self.high - self.low) / 2 + self.low
            else:
                evals = nodes
            
            evals = torch.tensor(evals)
            T = np.polynomial.Chebyshev.fit(nodes, self.fn(evals), self.degree)
            self.set_coeffs(T.coef.tolist())
            self.set_depth()

    def set_coeffs(self, coeffs):
        self.coeffs = coeffs

    def set_depth(self):
        self.depth = int(math.ceil(math.log2(self.degree+1)))
        if self.prescale != 1: # additional level needed
            self.depth += 1

    def set_output_scale(self, output_scale):
        self.output_scale = output_scale

    def compile(self):
        self.poly = self.scheme.poly_evaluator.generate_chebyshev(self.coeffs)

    @timer
    def forward(self, x):  
        if not self.he_mode:
            return self.fn(x)

        # Scale into [-1, 1] if needed.
        if not self.fused:
            if self.prescale != 1:
                x *= self.prescale 
            if self.constant != 0:
                x += self.constant

        return self.scheme.poly_evaluator.evaluate_polynomial(
            x, self.poly, self.output_scale)
    

class ELU(Chebyshev):
    def __init__(self, alpha=1.0, degree=31):
        self.alpha = alpha
        super().__init__(degree, self.fn) 
        
    def fn(self, x):
        return torch.where(x > 0, x, self.alpha * (torch.exp(x) - 1))
    

class Hardshrink(Chebyshev): 
    def __init__(self, degree=31, lambd=0.5):
        self.lambd = lambd
        super().__init__(degree, self.fn) 
    
    def fn(self, x):
        return torch.where((x > self.lambd) | (x < -self.lambd), x, torch.tensor(0.0))
    

class GELU(Chebyshev): 
    def __init__(self, degree=31):
        super().__init__(degree, self.fn) 
    
    def fn(self, x):
        return F.gelu(x)
    

class SiLU(Chebyshev):
    def __init__(self, degree=31):
        super().__init__(degree, self.fn) 

    def fn(self, x):
        return F.silu(x)
    

class Sigmoid(Chebyshev):
    def __init__(self, degree=31):
        super().__init__(degree, self.fn) 
        
    def fn(self, x):
        return F.sigmoid(x)
    

class SELU(Chebyshev):
    def __init__(self, degree=31):
        super().__init__(degree, self.fn) 
        
    def fn(self, x):
        alpha = 1.6732632423543772
        scale = 1.0507009873554805
        return scale * torch.where(x > 0, x, alpha * (torch.exp(x) - 1))
    

class Softplus(Chebyshev):
    def __init__(self, degree=31):
        super().__init__(degree, self.fn) 
        
    def fn(self, x):
        return F.softplus(x)
    

class Mish(Chebyshev):
    def __init__(self, degree=31):
        super().__init__(degree, self.fn) 
        
    def fn(self, x):
        return x * torch.tanh(F.softplus(x))
    

class _Sign(Module):
    def __init__(
        self, 
        degrees=[15,15,27],
        prec=128,
        logalpha=6,
        logerr=12,
    ):
        super().__init__()
        self.degrees = degrees
        self.prec = prec 
        self.logalpha = logalpha 
        self.logerr = logerr 
        self.mult = Mult()

        acts = []
        for i, degree in enumerate(degrees):
            is_last = (i == len(degrees) - 1)
            fn = self.fn1 if not is_last else self.fn2
            act = Chebyshev(degree, fn, within_composite=True)
            acts.append(act)
        
        self.acts = nn.Sequential(*acts)

    def extra_repr(self):
        return super().extra_repr() + f", degrees={self.degrees}"
            
    def fit(self):
        debug = self.scheme.params.get_debug_status()
        self.coeffs = self.scheme.poly_evaluator.generate_minimax_sign_coeffs(
            self.degrees, self.prec, self.logalpha, self.logerr, debug)
        
        for i, coeffs in enumerate(self.coeffs):
            self.acts[i].set_coeffs(coeffs)
            self.acts[i].set_depth()
                
    def fn1(self, x):
        return torch.where(x <= 0, torch.tensor(-1.0), torch.tensor(1.0))
    
    def fn2(self, x):
        return torch.where(x <= 0, torch.tensor(0.0), torch.tensor(1.0))

    def forward(self, x):
        if self.he_mode:
            l1 = x.level() 
            l2 = self.acts[-1].level - self.acts[-1].depth 
            
            # We'll calculate the output level of sign on the fly by 
            # comparing and taking the minimum of x and sign(x), as FHE
            # multiplication will do the same. Then, we'll set the output
            # scale of sign to be the modulus in the chain at this level.
            # This way, rescaling divides ql / ql and is exact.
            output_level = min(l1, l2)
            ql = self.scheme.encoder.get_moduli_chain()[output_level]
            self.acts[-1].set_output_scale(ql)

        # Composite polynomial evaluation
        for act in self.acts:
            x = act(x)
        return x


class ReLU(Module):
    def __init__(self, 
                 degrees=[15,15,27],
                 prec=128,
                 logalpha=6,
                 logerr=12,
    ):
        super().__init__()
        self.degrees = degrees 
        self.prec = prec 
        self.logalpha = logalpha
        self.logerr = logerr 
        self.sign = _Sign(degrees, prec, logalpha, logerr)
        self.mult1 = Mult()
        self.mult2 = Mult()

        self.prescale = 1 
        self.postscale = 1

    def extra_repr(self):
        return super().extra_repr() + f", degrees={self.degrees}"
    
    def fit(self):
        self.input_min = self.mult1.input_min 
        self.input_max = self.mult1.input_max

        absmax = max(abs(self.input_min), abs(self.input_max)) * self.margin
        if absmax > 1:
            self.postscale = int(math.ceil(absmax))
            self.prescale = 1 / self.postscale
    
    @timer
    def forward(self, x):
        x = self.mult1(x, self.prescale)
        x = self.mult2(x, self.sign(x))
        x *= self.postscale # integer mult, no level consumed
        return x


class Cleanse(Module):
    def __init__(self, iter=3):
        super().__init__()
        self.polys = nn.Sequential(*[Activation(coeffs=[-2, 3, 0, 0]) for _ in range(iter)])

    def forward(self, x):
        for poly in self.polys:
            x = poly(x)
        return x


class SqMethod1(Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        self.poly = Activation(coeffs=[-2/self.p**2, 0, 1])

    def forward(self, x):
        return self.poly(x)


class SqMethod2(Module):
    def __init__(self, r):
        super().__init__()
        self.r = r
        self.quad = nn.Sequential(*[Quad() for _ in range(self.r)])

    def forward(self, x):
        for quad in self.quad:
            x = quad(x)
        return x


class SqMethod(Module):
    def __init__(self, p, r):
        super().__init__()
        self.p = p
        self.r = r
        self.sq1 = SqMethod1(p)
        self.sq2 = SqMethod2(r)

    def forward(self, x):
        x = self.sq1(x)
        x = self.sq2(x)
        return x


class EIF(Module):
    def __init__(self, p, r, iter):
        super().__init__()
        self.p = p
        self.r = r
        self.iter = iter
        self.cleanse = Cleanse(iter)
        self.sq = SqMethod(p, r)

    def forward(self, x):
        x = self.sq(x)
        x = self.cleanse(x)
        return x