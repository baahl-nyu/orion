import sys
import math
from abc import abstractmethod

import torch
import torch.nn as nn

from .module import Module, timer
from ..core import packing


class LinearTransform(Module):
    def __init__(self, bsgs_ratio, level) -> None:
        super().__init__()
        self.bsgs_ratio = float(bsgs_ratio)
        self.set_depth(1)
        self.set_level(level)

        self.diagonals = {} # diags[(row, col)] = {0: [...], 1: [...], ...}
        self.transform_ids = {} # ids[(row, col)] = int
        self.output_rotations = 0

    def __del__(self):
        if 'sys' in globals() and sys.modules and self.scheme:
            try:
                self.scheme.lt_evaluator.delete_transforms(self.transform_ids)
            except Exception:
                pass # avoids errors for GC at program termination

    def extra_repr(self):
        return super().extra_repr() + f", bsgs_ratio={self.bsgs_ratio}"
            
    def init_orion_params(self):
        # Initialize additional Orion-specific weights/biases.
        self.on_weight = self.weight.data.clone()
        self.on_bias = (self.bias.data.clone() if hasattr(self, 'bias') and 
                        self.bias is not None else torch.zeros(self.weight.shape[0]))

    @abstractmethod
    def compute_fhe_output_gap(self, **kwargs):
        """Compute the multiplexed output gap."""
        pass

    @abstractmethod
    def compute_fhe_output_shape(self, **kwargs) -> tuple:
        """Compute the FHE output dimensions after multiplexing."""
        pass

    @abstractmethod
    def generate_diagonals(self, last):
        # Here, we'll apply our packing strategies to return the diagonals
        # of our linear layer. When using the "hybrid" method of packing, this
        # may also require several output rotations and summations.

        # When ``io_mode`` is set to "load", the diagonals were already saved
        # to disk in a previous compilation. In that case, skip regeneration
        # here and allow ``compile()`` to load them directly from disk.
        if self.get_io_mode() == "load":
            return

    def get_io_mode(self):
        return self.scheme.params.get_io_mode()

    def save_transforms(self):
        self.scheme.lt_evaluator.save_transforms(self)

    def load_transforms(self):
        return self.scheme.lt_evaluator.load_transforms(self) 

    def compile(self):
        self.transform_ids = self.scheme.lt_evaluator.generate_transforms(self)

    @timer
    def evaluate_transforms(self, x):
        out = self.scheme.lt_evaluator.evaluate_transforms(self, x)

        # Hybrid method's output rotations
        slots = self.scheme.params.get_slots()
        for i in range(1, self.output_rotations+1):
            out += out.roll(slots // (2**i))

        out += self.on_bias_ptxt
        return out


class Linear(LinearTransform):    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True,
        bsgs_ratio: int = 2,
        level: int = None,
    ) -> None:
        super().__init__(bsgs_ratio, level)

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, " + 
                super().extra_repr())

    def reset_parameters(self):
        # Initialize weights and biases following standard PyTorch instantiation.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def compute_fhe_output_gap(self, **kwargs):
        return 1 # linear layers in reset the multiplexed gap to 1.

    def compute_fhe_output_shape(self, **kwargs) -> tuple:
        # Linear layers also remove any padded zeros, Therefore the output 
        # shape under FHE inference is identical to cleartext inference. 
        return kwargs["clear_output_shape"]
        
    def generate_diagonals(self, last):
        # Here, we'll apply our packing strategies to return the diagonals
        # of our linear layer. When using the "hybrid" method of packing, this
        # may also require several output rotations and summations.
        self.diagonals, self.output_rotations = packing.pack_linear(self, last)
        if self.get_io_mode() == "save":
            self.save_transforms()

    def compile(self):
        # If the user specifies an I/O mode = "save" or "load", then diagonals will
        # be temporarily stored to disk to save memory. Load right before they're 
        # needed to generate the backend transforms themselves. 
        if self.get_io_mode() != "none":
            self.diagonals, self.on_bias, self.output_rotations = self.load_transforms()

        # We delay constructing the bias until now, so that any fusing can 
        # modify the bias variable beforehand.
        bias = packing.construct_linear_bias(self)
        self.on_bias_ptxt = self.scheme.encoder.encode(bias, self.level-self.depth)
        self.transform_ids = self.scheme.lt_evaluator.generate_transforms(self)
    
    def forward(self, x):
        if not self.he_mode:
            if x.dim() != 2:
                extra = " Forgot to call on.Flatten() first?" if x.dim() == 4 else ""
                raise ValueError(
                    f"Expected input to {self.__class__.__name__} to have "
                    f"2 dimensions (N, in_features), but got {x.dim()} " 
                    f"dimension(s): {x.shape}." + extra
        )
            
            # If we're not in FHE inference mode, then we'll just return
            # the default PyTorch result.
            return torch.nn.functional.linear(x, self.weight, self.bias)
        
        # Otherwise, call parent evaluation for FHE.
        return self.evaluate_transforms(x) 


class Conv2d(LinearTransform):    
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            kernel_size: int, 
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True,
            bsgs_ratio: int = 2,
            level: int = None,
    ) -> None:
        super().__init__(bsgs_ratio, level)

        # Standard PyTorch Conv2d attributes
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Convert int parameters to tuples
        self.kernel_size = self._make_tuple(kernel_size)
        self.stride = self._make_tuple(stride)
        self.padding = self._make_tuple(padding)
        self.dilation = self._make_tuple(dilation)
        self.groups = groups
        
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def _make_tuple(self, value):
        return (value, value) if isinstance(value, int) else value

    def reset_parameters(self):
        """Initialize parameters using PyTorch's standard approach."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if self.bias is not None:
            fan_in = self.weight.size(1) * self.weight.size(2) * self.weight.size(3)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        return (f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, "
                f"padding={self.padding}, dilation={self.dilation}, "
                f"groups={self.groups}, " + super().extra_repr())

    def compute_fhe_output_gap(self, **kwargs):
        # Strided convolutions increase the multiplexed gap by a factor 
        # of the stride.
        input_gap = kwargs['input_gap']  
        return input_gap * self.stride[0]
    
    def compute_fhe_output_shape(self, **kwargs) -> tuple:
        input_shape = kwargs['input_shape']
        clear_output_shape = kwargs['clear_output_shape']
        input_gap = kwargs['input_gap']

        Hi, Wi = input_shape[2:]
        N, Co, Ho, Wo = clear_output_shape
        output_gap = self.compute_fhe_output_gap(input_gap=input_gap)
        
        on_Co = math.ceil(Co / (output_gap**2))
        on_Ho = max(Hi, Ho*output_gap)
        on_Wo = max(Wi, Wo*output_gap)

        return torch.Size((N, on_Co, on_Ho, on_Wo))
    
    def generate_diagonals(self, last):
        # Generate Toeplitz diagonals and determine the number of output
        # rotations if the `hybrid` packing method is used.
        self.diagonals, self.output_rotations = packing.pack_conv2d(self, last)
        if self.get_io_mode() == "save":
            self.save_transforms()

    def compile(self):
        # If the user specifies an io mode = "save" or "load", then diagonals will
        # be temporarily stored to disk to save memory. Load right before they're 
        # needed to generate the backend transforms themselves. 
        if self.get_io_mode() != "none":
            self.diagonals, self.on_bias, self.output_rotations = self.load_transforms()

        # We delay constructing the bias until now, so that any fusing can 
        # modify the bias variable beforehand.
        bias = packing.construct_conv2d_bias(self)
        self.on_bias_ptxt = self.scheme.encoder.encode(bias, self.level-self.depth)
        self.transform_ids = self.scheme.lt_evaluator.generate_transforms(self)

    def forward(self, x):
        # Forward pass that handles both cleartext and FHE inference.
        if not self.he_mode: # cleartext mode
            if x.dim() != 4:
                raise ValueError(
                    f"Expected input to {self.__class__.__name__} to have "
                    f" 4 dimensions (N, C, H, W), but got {x.dim()} "
                    f"dimension(s): {x.shape}."
                )
            return torch.nn.functional.conv2d(
                x, self.weight, self.bias, self.stride, 
                self.padding, self.dilation, self.groups
            )
        
        return self.evaluate_transforms(x) # FHE mode