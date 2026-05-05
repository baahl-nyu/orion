import math 
import torch
import torch.nn as nn

from .module import Module
from .linear import Linear

from orion.backend.python.tensors import CipherTensor

class Extract(Module):
    """
    Extracts a slice of a 1-d ciphertensor
    numel: total number of elements in the input tensor
    extracted_size: number of elements to extract
    offset: offset of the slice
    """
    def __init__(self, numel, extracted_size, offset):
        super().__init__()
        self.set_depth(1)
        self.numel = numel
        self.extracted_size = extracted_size
        self.offset = offset

    def compile(self):
        self.mask = torch.zeros(self.numel)
        self.mask[self.offset:self.offset+self.extracted_size] = 1.0
        q1 = self.scheme.encoder.get_moduli_chain()[self.level]
        self.mask_ptxt = self.scheme.encoder.encode(self.mask, self.level, q1)
        self.scheme.evaluator.add_rotation_key(self.offset)
    
    def forward(self, x):
        if not self.he_mode:
            out = x[..., self.offset:self.offset+self.extracted_size]
            return out

        # otherwise, we are in FHE mode
        x = x * self.mask_ptxt
        x = x.roll(self.offset)
        x.on_shape = torch.Size([1, self.extracted_size])
        x.shape = torch.Size([1, self.extracted_size])
        return x
    

class ExtractLinear(Linear):
    """
    Extracts a slice of a 1-d ciphertensor using a linear transformation
    numel: total number of elements in the input tensor
    extracted_size: number of elements to extract
    offset: offset of the slice

    example:
    x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    extract = ExtractLinear(8, 4, 0)
    out = extract(x)
    print(out)
    # tensor([1, 2, 3, 4])

    extract = Extract(8, 4, 4)
    out = extract(x)
    print(out)
    # tensor([5, 6, 7, 8])
    """
    def __init__(self, numel, extracted_size, offset):
        super().__init__(numel, extracted_size, bias=False)
        self.numel = numel
        self.extracted_size = extracted_size
        self.offset = offset

        self.weight.data[:] = 0.
        self.weight.data[:self.extracted_size, self.offset:self.offset+self.extracted_size] = torch.eye(self.extracted_size)

    def forward(self, x):
        return super().forward(x)
    

class ExtractSparse(Module):
    """
    important: This is very specific layer only meant for large DLRM models using the Criteo Dataset.
    """
    def __init__(self, num_dense, vocab_size, slots=32768):
        super().__init__()
        self.set_depth(1)
        self.num_dense = num_dense
        self.slots = slots
        self.vocab_size = vocab_size
        self.num_ctxts = math.ceil((self.vocab_size+self.num_dense) / self.slots)
        #print(f"num_ctxts: {self.num_ctxts}")

    def compile(self):
        # 
        self.mask0 = torch.zeros(self.slots * self.num_ctxts)
        for i in range(1, self.num_ctxts):
            self.mask0[i*self.slots:i*self.slots+self.num_dense] = 1.0

        # 
        self.mask1 = torch.ones(self.slots * self.num_ctxts)
        for i in range(self.num_ctxts):
            self.mask1[i*self.slots:i*self.slots+self.num_dense] = 0.0

        q1 = self.scheme.encoder.get_moduli_chain()[self.level]
        self.mask_ptxt0 = self.scheme.encoder.encode(self.mask0, self.level, q1)
        self.mask_ptxt1 = self.scheme.encoder.encode(self.mask1, self.level, q1)
        self.scheme.evaluator.add_rotation_key(self.num_dense)
    
    def forward(self, x):
        if not self.he_mode:
            out = x[..., self.num_dense:]
            return out

        # otherwise, we are in FHE mode
        # per ciphertext FHE ops, be aware! again, this implementation
        # is very specific to the HE-LRM project.
        x.on_shape = torch.Size([1, self.slots * self.num_ctxts])

        out_mask0 = x * self.mask_ptxt0
        out_mask1 = x * self.mask_ptxt1

        out_mask0 = out_mask0.roll(self.num_dense)
        out_mask1 = out_mask1.roll(self.num_dense)

        ids = []
        for i in range(1, len(out_mask0.ids)):
            ids.append(out_mask0.ids[i])
        ids.append(out_mask0.ids[0])

        out_size = out_mask0.on_shape
        out = CipherTensor(self.scheme, ids, out_size, out_size)

        final = out + out_mask1
        final.on_shape = torch.Size([1, self.vocab_size])

        return final

