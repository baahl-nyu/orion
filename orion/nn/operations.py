import math
import torch

from .module import Module, timer

class Add(Module):
    def __init__(self):
        super().__init__()
        self.set_depth(0)

    def compute_fhe_output_shape(self, **kwargs):
        fhe_input_shape = kwargs["fhe_input_shape"]

        if isinstance(fhe_input_shape, list):
            return fhe_input_shape[0]
        return fhe_input_shape

    def forward(self, x, y):
        return x + y
    

class Mult(Module):
    def __init__(self):
        super().__init__()
        self.set_depth(1)

    def compute_fhe_output_shape(self, **kwargs):
        fhe_input_shape = kwargs["fhe_input_shape"]

        if isinstance(fhe_input_shape, list):
            return fhe_input_shape[0]
        return fhe_input_shape

    def forward(self, x, y):
        return x * y
    

class Bootstrap(Module):
    def __init__(self, input_min, input_max, input_level):
        super().__init__()
        self.input_min = input_min 
        self.input_max = input_max 
        self.input_level = input_level
        self.prescale = 1
        self.postscale = 1
        self.constant = 0

    def extra_repr(self):
        l_eff = len(self.scheme.params.get_logq()) - 1
        return f"l_eff={l_eff}"

    def fit(self):
        center = (self.input_min + self.input_max) / 2 
        half_range = (self.input_max - self.input_min) / 2
        self.low = (center - (self.margin * half_range)).item()
        self.high = (center + (self.margin * half_range)).item()

        # We'll want to scale from [A, B] into [-1, 1] using a value of the
        # form 1 / integer, so that way our multiplication back to the range
        # [A, B] (by integer) after bootstrapping doesn't consume a level.
        if self.high - self.low > 2:
            self.postscale = math.ceil((self.high - self.low) / 2)
            self.prescale = 1 / self.postscale

        self.constant = -(self.low + self.high) / 2 

    def compile(self):
        # We'll then encode the prescale at the level of the input ciphertext
        # to ensure its rescaling is errorless
        elements = self.fhe_input_shape.numel()
        curr_slots = 2 ** math.ceil(math.log2(elements))

        prescale_vec = torch.zeros(curr_slots)
        prescale_vec[:elements] = self.prescale

        ql = self.scheme.encoder.get_moduli_chain()[self.input_level]
        self.prescale_ptxt = self.scheme.encoder.encode(
            prescale_vec, level=self.input_level, scale=ql)

    @timer
    def forward(self, x):
        if not self.he_mode:
            return x
        
        # Shift and scale into range [-1, 1]. Important caveat -- here we first
        # shift, then scale. This let's us zero out unused slots and enables
        # sparse bootstrapping (i.e., where slots < N/2).
        if self.constant != 0:
            x += self.constant
        x *= self.prescale_ptxt
 
        x = x.bootstrap()

        # Scale and shift back to the original range
        if self.postscale != 1:
            x *= self.postscale 
        if self.constant != 0:
            x -= self.constant

        return x




