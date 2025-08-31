import math
import torch
import opt_einsum
import torch.nn as nn

from .module import Module
from .linear import Linear

""" 
See https://github.com/baahl-nyu/einhops/tree/main for the original implementation.
"""

class Einsum(Module):
    def __init__(self, equation):
        super().__init__()
        self.equation = equation
        self.shapes = None
        self.num_inputs = self._get_num_inputs()
        self._validate_num_inputs()

        # instantiate linear transforms
        self.slots_ready = False
        self.shapes_ready = False
        self.LTs_ready = False
        self.broadcast_rotations_ready = False
        self.reduction_rotations_ready = False
        self.LTs = nn.Sequential(*[Linear(1, 1, bias=False) for _ in range(self._get_num_inputs() + 1)])

    def extra_repr(self):
        return super().extra_repr() + f", einsum({self.equation})"
    
    def _update_shapes(self, out, *args):
        """
        Capture the cleartext shapes of the inputs and output
        for this particular einsum circuit.
        """
        
        if self.shapes_ready:
            return

        self.plain_output_shape = out.shape
        self.plain_input_shapes = [arg.shape for arg in args]
        self.shapes_ready = True

    def _update_slots(self):
        """
        Capture the number of slots.
        """
        if self.slots_ready:
            return
        
        self.slots = self.scheme.params.get_slots()
        self.slots_ready = True

    def _validate_slot_usage(self):
        """
        Make sure that the number of slots is sufficient for the einsum circuit.
        """
        num_slots_required = math.prod(self.po2_dim_sizes.values())
        if num_slots_required > self.slots:
            raise ValueError("Total number of slots ({num_slots_required}) exceeds the maximum number of slots ({self.slots})")
        
    def _validate_num_inputs(self):
        """
        Validate that the number of inputs is correct.
        """
        if self.num_inputs > 2:
            raise ValueError(f"Einsum supports at most 2 inputs, but {self.num_inputs} were provided.")

    def _get_dim_sizes(self, input_dims, shapes):
        """
        Associate each labeled dimension with its size
        """
        dim_sizes = {}
        for input_dim, shape in zip(input_dims, shapes):
            for i, dim in enumerate(input_dim):
                dim_sizes[dim] = shape[i]
        return dim_sizes
    
    def _get_po2_dim_sizes(self, input_dims, shapes):
        """
        Associate each labeled dimension with the next power of two
        greater than or equal to its size. We need this for
        utilizing the rotation-and-summation method.
        """
        next_power_of_two = lambda n: 1 << ((n - 1).bit_length())
        po2_dim_sizes = {}
        for input_dim, shape in zip(input_dims, shapes):
            for i, dim in enumerate(input_dim):
                po2_dim_sizes[dim] = next_power_of_two(shape[i])
        return po2_dim_sizes

    def _parse_dims(self, shapes):
        """
        Parse the dimensions from the equation using opt_einsum.
        """
        input_subs, o_dims, _ = opt_einsum.parser.parse_einsum_input((self.equation, *[torch.empty(shape) for shape in shapes]))
        i_dims = input_subs.strip().split(",")

        seen = set()
        r_dims = "".join(
            d for d in "".join(i_dims)
            if d not in o_dims and not (d in seen or seen.add(d))
        )
        return i_dims, o_dims, r_dims
    
    def _calculate_new_idxs(self, src_dim, dst_dim, dim_sizes, po2_dim_sizes):
        """
        Uses strides to map src values to their new locations in dst.
        """
        src_shape = [dim_sizes[dim] for dim in src_dim]
        src_stride = torch.tensor(torch.empty(*src_shape).stride())

        # get the stride of each dim in the dst tensor
        dst_shape = [po2_dim_sizes[dim] for dim in dst_dim]
        dst_stride = torch.tensor(torch.empty(*dst_shape).stride())

        # map src dimensions to their new stride in dst
        src_to_dst_indices = [dst_dim.index(d) for d in src_dim]
        new_stride = dst_stride[src_to_dst_indices]

        # get flattened indices of src tensor
        src_tensor_idxs = torch.cartesian_prod(*[torch.arange(dim_sizes[dim]) for dim in src_dim])
        if src_tensor_idxs.ndim == 1: # handling 1-d arrays
            src_tensor_idxs = src_tensor_idxs.unsqueeze(1)
        src_idxs = torch.sum(src_tensor_idxs * src_stride, dim=1)
        new_idxs = torch.sum(src_tensor_idxs * new_stride, dim=1)
        return new_idxs, src_idxs
    
    def _update_LTs(self, shapes):
        """
        Generates the linear transforms required for the einsum circuit and prepares
        the broadcast rotations.
        """
        if self.LTs_ready:
            return
        
        if not (self.shapes_ready and self.slots_ready):
            raise ValueError("Shapes and slots must be updated before updating LT.")
        
        self.i_dims, self.o_dims, self.r_dims = self._parse_dims(shapes)
        broadcasted_dims = self.r_dims + self.o_dims # each input will be broadcasted to this shape

        self.dim_sizes = self._get_dim_sizes(self.i_dims, shapes)
        self.po2_dim_sizes = self._get_po2_dim_sizes(self.i_dims, shapes)

        broadcast_shape = [self.po2_dim_sizes[dim] for dim in broadcasted_dims]
        broadcasted_strides = torch.tensor(torch.empty(broadcast_shape).stride())
        self.broadcast_rots = []

        for i, (src_dim, shape) in enumerate(zip(self.i_dims, shapes)):
            # Step 1: Calculate the permutation matrix needed for each input.
            new_idxs, src_idxs = self._calculate_new_idxs(src_dim, broadcasted_dims, self.dim_sizes, self.po2_dim_sizes)
            ni, no = self.plain_input_shapes[i].numel(), self.slots
            self.LTs[i].in_features = ni
            self.LTs[i].out_features = no
            W = torch.zeros(no, ni)
            W[new_idxs, src_idxs] = 1.0
            self.LTs[i].weight = nn.Parameter(W)

            # Step 2: Calculate the rotations needed for each input.
            self.broadcast_rots.append([])
            for dim in reversed(broadcasted_dims):
                if dim not in src_dim:
                    stride = broadcasted_strides[broadcasted_dims.index(dim)].item()
                    num_rots = int(math.log2(self.po2_dim_sizes[dim]))
                    self.broadcast_rots[i].append((stride, num_rots))

        # Step 3: Calculate the linear transform needed for the output.
        ni, no = self.slots, self.plain_output_shape.numel()
        self.LTs[-1].in_features = ni
        self.LTs[-1].out_features = no
        W = torch.zeros(no, ni)
        if len(self.o_dims) > 0:
            src_idxs, dst_idxs = self._calculate_new_idxs(self.o_dims, self.o_dims, self.dim_sizes, self.po2_dim_sizes)
            W[dst_idxs, src_idxs] = 1.0
        else:
            W[0, 0] = 1.0 # there's only a singleton output, zero all other slots out.
        self.LTs[-1].weight = nn.Parameter(W)

        self.LTs_ready = True

    def _update_reduction_rotations(self):
        """
        Calculate the amount of reduction rotations needed for the output.
        """
        if self.reduction_rotations_ready:
            return
        
        self.reduction_rot_amount = math.prod([self.po2_dim_sizes[dim] for dim in self.o_dims])
        num_repeated_o_dims = math.prod([self.po2_dim_sizes[dim] for dim in self.r_dims])
        self.num_reduction_rots = int(math.log2(num_repeated_o_dims))

        self.reduction_rotations_ready = True

    def _is_fitting(self, *args):
        """
        Check if the inputs are torch.fx.proxy.Proxy.
        """
        return isinstance(args[0], torch.fx.proxy.Proxy)

    def _flatten_inputs(self, *args):
        """
        Flatten the inputs to prepare for the einsum circuit.
        """
        for arg in args:
            arg.on_shape = torch.Size([1, arg.shape.numel()])
            arg.shape = torch.Size([1, arg.shape.numel()])

    def _assign_final_shape(self, out):
        """
        Assign the final shape to the output.
        """
        out.on_shape = self.plain_output_shape
        out.shape = self.plain_output_shape

    def _get_num_inputs(self):
        """
        Get the number of inputs from the equation.
        """
        return len(self.equation.split(","))
    
    def _plain_forward(self, *args):
        """
        Forward pass for plaintext inference. Inputs are torch.Tensors.
        Shapes, slots, and the required linear transforms are captured. 
        """

        if "..." in self.equation:
            raise ValueError("Ellipsis (...) is not supported in einsum expressions.")
        
        if "->" not in self.equation:
            raise ValueError("Equation must contain '->'.")
        out = torch.einsum(self.equation, *args)

        # prepare for FHE circuit
        self._update_shapes(out, *args)
        self._update_slots()
        self._update_LTs([arg.shape for arg in args])
        self._update_reduction_rotations()
        self._validate_slot_usage()
        return out
    
    def _trace_forward(self, *args):
        """
        Forward pass for symbolic trace. Inputs are torch.fx.proxy.Proxy.
        This pass must emulate the FHE einsuum circuit.
        """

        # Step 1: Perform the broadcast
        ctxts = []
        for i in range(self._get_num_inputs()):
            ctxts.append(self.LTs[i](args[i].view(1, -1)))
            for (stride, num_rots) in self.broadcast_rots[i]:
                curr_stride = stride
                curr_num_rots = num_rots
                for j in range(curr_num_rots):
                    ctxts[i] += ctxts[i].roll(+curr_stride)
                    curr_stride *= 2


        # Step 2: Perform the mult
        if len(ctxts) == 1:
            out3 = ctxts[0]
        else: 
            out3 = ctxts[0] * ctxts[1]

        # Step 3: Perform the reduction
        rot_amount = self.reduction_rot_amount
        for i in range(self.num_reduction_rots):
            out3 += out3.roll(-rot_amount)
            rot_amount *= 2
        out4 = self.LTs[-1](out3)
        return out4
    
    def _fhe_forward(self, *args):
        """
        Forward pass for FHE inference. Inputs are ciphertexts.
        """
        self._flatten_inputs(*args)

        # Step 1: Perform the broadcast
        ctxts = []
        for i in range(self._get_num_inputs()):
            ctxts.append(self.LTs[i](args[i]))
            for (stride, num_rots) in self.broadcast_rots[i]:
                curr_stride = stride
                curr_num_rots = num_rots
                for j in range(curr_num_rots):
                    rotated = ctxts[i].roll(-curr_stride)
                    ctxts[i] = ctxts[i] + rotated
                    curr_stride *= 2

        # Step 2: Perform the mult
        if len(ctxts) == 1:
            out3 = ctxts[0]
        else: 
            out3 = ctxts[0] * ctxts[1]

        out3.set_scale(args[0].scale()) # manually set the scale

        # Step 3: Perform the reduction
        rot_amount = self.reduction_rot_amount
        for i in range(self.num_reduction_rots):
            rotated = out3.roll(rot_amount)
            out3 = out3 + rotated
            rot_amount *= 2
        out4 = self.LTs[-1](out3)

        self._assign_final_shape(out4)
        return out4

    def forward(self, *args):
        if self.he_mode:
            return self._fhe_forward(*args)
        elif self._is_fitting(*args):
            return self._trace_forward(*args)
        return self._plain_forward(*args)
    