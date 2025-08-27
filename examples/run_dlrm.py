import time
import math

import torch
import torch.nn as nn 
import torch.nn.functional as F

import orion
import orion.nn as on
from orion.core.utils import mae

orion.set_log_level('DEBUG')


class DLRM(on.Module):
    def __init__(self, dense_size, vocab_sizes, hidden_dim):
        super().__init__()
        
        self.dense_size = dense_size
        self.vocab_sizes = vocab_sizes
        self.hidden_dim = hidden_dim
        self.num_sparse = len(vocab_sizes)  
        
        # Bottom MLP - Process dense features
        self.bot_l = nn.Sequential(
            on.Linear(self.dense_size, 128),
            on.ReLU(),
            on.Linear(128, self.hidden_dim),
            on.ReLU(),
        )
        
        # Embedding tables - Process sparse features
        self.emb_l = nn.ModuleList()
        for vocab_size in self.vocab_sizes:
            self.emb_l.append(on.Embedding(vocab_size, self.hidden_dim))
        
        # Top MLP - Process interactions
        self.top_l = nn.Sequential(
            on.Linear(self.hidden_dim, 128),
            on.ReLU(),
            on.Linear(128, 128),
            on.ReLU(),
            on.Linear(128, 1),
        )
    
    def forward(self, x, sparse_inputs):  
        bot = self.bot_l(x)  

        emb = 0
        for i in range(self.num_sparse):
            emb += self.emb_l[i](sparse_inputs[i])  
       
        interact = emb + bot
        return self.top_l(interact)


if __name__ == "__main__":
    torch.manual_seed(0)
   
    dense_size  = 128
    vocab_sizes = [128, 128, 128, 128, 128]
    hidden_dim  = 128

    dlrm = DLRM(dense_size, vocab_sizes, hidden_dim)
    dlrm.eval()

    # Create example dense and sparse features
    dense = torch.randn(1, dense_size)
    sparse_idxs = [torch.randint(0, size, (1,)) for size in vocab_sizes]
    sparse = [F.one_hot(idx, size).float() for idx, size in zip(sparse_idxs, vocab_sizes)]

    # Initialize Orion
    scheme = orion.init_scheme("../configs/resnet.yml")

    # Forward pass in the clear
    out_clear = dlrm(dense, sparse)

    orion.fit(dlrm, input_data=[dense, sparse])
    input_level = orion.compile(dlrm)

    # Encode and encrypt the input vector 
    dense_ctxt = orion.encrypt(orion.encode(dense, input_level))
    sparse_ctxts = [orion.encrypt(orion.encode(s, input_level)) for s in sparse]

    dlrm.he()  # Switch to FHE mode

    # Run FHE inference
    print("\nStarting FHE inference", flush=True)
    start = time.time()
    out_ctxt = dlrm(dense_ctxt, sparse_ctxts)
    end = time.time()

    # Get the FHE results and decrypt + decode.
    out_ptxt = out_ctxt.decrypt()
    out_fhe = out_ptxt.decode()

    # Compare the cleartext and FHE results.
    print()
    print(out_clear)
    print(out_fhe)

    dist = mae(out_clear, out_fhe)
    print(f"\nMAE: {dist:.4f}")
    print(f"Precision: {-math.log2(dist):.4f}")
    print(f"Runtime: {end-start:.4f} secs.\n")