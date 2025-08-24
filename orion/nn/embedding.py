import math 
import torch
import torch.nn as nn

from .module import Module
from .linear import Linear
from .activation import EIF

from orion.backend.python.tensors import CipherTensor


class Embedding(Linear):
    """
    Embedding layer that uses a lookup table to store the embedding vectors.
    Input is a one-hot encoded vector.
    """
    def __init__(self, vocab_size, embedding_dim):
        super().__init__(vocab_size, embedding_dim, bias=False)
    
    def forward(self, x):
        return super().forward(x)


class CompressedEmbedding(Linear):
    """
    Concatenated compressed embedding tables (https://arxiv.org/abs/1711.01068)
    where the input is l concatenated one-hot encoded vectors.
    """
    def __init__(self, p, l, embedding_dim):
        super().__init__(p*l, embedding_dim, bias=False)

    def forward(self, x):
        return super().forward(x)


class CompressedIndicatorEmbedding(Module):
    """
    Cocompressed embedding tables (https://arxiv.org/abs/1711.01068)
    with indicator function must be applied to obtain the one-hot encoded vectors.
    """

    def __init__(self, p, l, r, iter, embedding_dim):
        super().__init__()

        self.EIF = EIF(p=p, r=r, iter=iter)
        self.EmbeddingTable = Linear(p*l, embedding_dim, bias=False)
        
    def forward(self, x): 
        x = self.EIF(x)
        x = self.EmbeddingTable(x)
        return x


class BaselineEmbedding(Module):
    """
    Baseline embedding used in:
    Privacy-Preserving Embedding via Look-up Table Evaluation with Fully 
    Homomorphic Encryption. See Table 4, Row 4 (Method CodedHELUT+p1)

    """
    def __init__(self, p, l, r, iter, embedding_dim, consolidate=False):
        super().__init__()

        self.p = p
        self.l = l
        self.r = r
        self.iter = iter
        self.embedding_dim = embedding_dim

        self.EIF = EIF(p=p, r=r, iter=iter)
        if consolidate:
            self.table_mult = TableMultConsolidate(
                num_tables=l, 
                vocab_size=p, 
                embedding_dim=embedding_dim
            )
        else:
            self.table_mult = TableMult(
                num_tables=l, 
                vocab_size=p, 
                embedding_dim=embedding_dim
            )
    
    def forward(self, x):
        x = self.EIF(x) # We assume the input is already masked.
        x = self.table_mult(x)
        return x # a list of ciphertexts where each ciphertext has a single hidden dimension.


class TableMult(Module):
    def __init__(self, num_tables, vocab_size, embedding_dim):
        super().__init__()
        self.set_depth(1)

        self.num_tables = num_tables
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty((num_tables, vocab_size, embedding_dim)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def compile(self):
        q1 = self.scheme.encoder.get_moduli_chain()[self.level]
        concat_tables = self.weight.view(-1, self.embedding_dim)
        self.compressed_tables_ptxts = []
        for i in range(self.embedding_dim):
            curr_ptxt = self.scheme.encoder.encode(concat_tables[..., i], self.level, q1)
            self.compressed_tables_ptxts.append(curr_ptxt)

    def forward(self, x):
        if not self.he_mode:
            return (x * self.weight.view(-1, self.embedding_dim).T).sum(dim=1)

        # otherwise, we are in FHE mode

        # O(d) mults where d is the embedding dimension
        table_mults = []
        for i in range(len(self.compressed_tables_ptxts)):
            table_mults.append(x * self.compressed_tables_ptxts[i])

        # O(d log(p*l)) rotations where d is the embedding dimension, p is the 
        # vocab size, and l is the number of tables this makes sure we sum up 
        # all the compressed table embedding values.
        num_rots = math.ceil(math.log2(self.vocab_size * self.num_tables))    
        for i in range(len(table_mults)):
            rot_amount = (self.vocab_size * self.num_tables) // 2
            for j in range(num_rots):
                table_mults[i] += table_mults[i].roll(rot_amount)
                rot_amount //= 2

        # returning a ciphertensor object for the sake of bootstrapping
        # added a new decrypt/decode method since the first slot in each 
        # separate ciphertext contains the data we need.
        ids = []
        slots = self.scheme.params.get_slots()
        for table_mult in table_mults:
            for i in range(len(table_mult.ids)):
                ids.append(table_mult.ids[i])
        
        return CipherTensor(
            self.scheme, ids, 
            shape=None, 
            on_shape=None, 
            start=0, 
            stride=slots, 
            stop=slots*self.embedding_dim
        )


# The baseline method places a single output value in a separate ciphertext.
# This method consolidates all the output values into a single ciphertext.
class TableMultConsolidate(Module):
    def __init__(self, num_tables, vocab_size, embedding_dim):
        super().__init__()
        self.set_depth(2)

        self.num_tables = num_tables
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty((num_tables, vocab_size, embedding_dim)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def compile(self):
        q1 = self.scheme.encoder.get_moduli_chain()[self.level]
        q2 = self.scheme.encoder.get_moduli_chain()[self.level - 1]

        concat_tables = self.weight.view(-1, self.embedding_dim)
        self.compressed_tables_ptxts = []
        for i in range(self.embedding_dim):
            curr_ptxt = self.scheme.encoder.encode(concat_tables[..., i], self.level, q1)
            self.compressed_tables_ptxts.append(curr_ptxt)

        # mask for consolidating the output ciphertexts into a single ciphertext
        slots = self.scheme.params.get_slots()
        mask = torch.zeros(slots)
        mask[0] = 1
        self.mask = self.scheme.encoder.encode(mask, self.level - 1, q2)

        for i in range(1, self.embedding_dim):
            self.scheme.evaluator.add_rotation_key(-i)


    def forward(self, x):
        if not self.he_mode:
            return (x * self.weight.view(-1, self.embedding_dim).T).sum(dim=1)

        # otherwise, we are in FHE mode

        # O(d) mults where d is the embedding dimension
        table_mults = []
        for i in range(len(self.compressed_tables_ptxts)):
            table_mults.append(x * self.compressed_tables_ptxts[i])

        # O(d log(p*l)) rotations where d is the embedding dimension, p is the 
        # vocab size, and l is the number of tables. This makes sure we sum up 
        # all the compressed table embedding values.
        num_rots = math.ceil(math.log2(self.vocab_size * self.num_tables))    
        for i in range(len(table_mults)):
            rot_amount = (self.vocab_size * self.num_tables) // 2
            for j in range(num_rots):
                table_mults[i] += table_mults[i].roll(rot_amount)
                rot_amount //= 2

        # combine all sparse ciphertexts into a dense ciphertext that contains the
        # embedding in the first self.embedding_dim slots.
        out = table_mults[0] * self.mask
        for i in range(1, self.embedding_dim):
            masked = table_mults[i] * self.mask
            out += masked.roll(-i)
        out.on_shape = torch.Size([1, self.embedding_dim])
        
        return out