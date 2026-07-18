import torch

from .module import Module

class Flatten(Module):
    def __init__(self):
        super().__init__()
        self.set_depth(0)
        self.preserve_input_shapes = True

    def extra_repr(self):
        return super().extra_repr() + ", start_dim=1"
    
    def compute_fhe_output_shape(self, **kwargs):
        batch = kwargs["input_shape"][0]
        features = kwargs["input_shape"][1:].numel()
        return torch.Size([batch, features])
    
    def forward(self, x):
        if self.he_mode:
            return x 
        return torch.flatten(x, start_dim=1)