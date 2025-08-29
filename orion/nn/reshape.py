import torch

from .module import Module

class Flatten(Module):
    def __init__(self):
        super().__init__()
        self.set_depth(0)
        self.preserve_input_shape = True

    def extra_repr(self):
        return super().extra_repr() + ", start_dim=1"
    
    def forward(self, x):
        if self.he_mode:
            return x 
        return torch.flatten(x, start_dim=1)