import time
import math

import torch
import torch.nn as nn

import orion
import orion.nn as on
from orion.core.utils import mae


class ToyDLRM(on.Module):
    def __init__(self, num_dense, num_sparse):
        super().__init__()

        self.num_dense = num_dense
        self.num_sparse = num_sparse
        self.hidden_dim = 2
        self.vocab_size = [num_sparse]

        # Bottom MLP - Process dense features
        self.bot_l = nn.Sequential(
            on.Linear(self.num_dense, 3),
            on.ReLU(),
            on.Linear(3, 2),
            on.ReLU(),
        )

        # Embedding tables - Process sparse features
        self.emb_l = on.Embedding(self.vocab_size[0], self.hidden_dim)

        # Interaction layer (simple addition here)
        self.add = on.Add()
        
        # Top MLP - Process interactions
        self.top_l = nn.Sequential(
            on.Linear(2, 4),
            on.ReLU(),
            on.Linear(4, 2),
            on.ReLU(),
            on.Linear(2, 1),
        )

    def forward(self, dense_x, sparse_x):
        dense_out = self.bot_l(dense_x)
        sparse_out = self.emb_l(sparse_x)

        sparse_out = sparse_out.roll(-self.num_dense)
        interact_out = self.add(dense_out, sparse_out)

        return self.top_l(interact_out)


def main():
    # Set seed for reproducibility
    torch.manual_seed(42)

    # Initialize the Orion scheme, model, and data
    orion.init_scheme("../configs/resnet.yml")
    net = ToyDLRM(num_dense=5, num_sparse=4)

    dense_x  = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    sparse_x = torch.tensor([[0.0, 0.0, 1.0, 0.0]])

    # Run cleartext inference
    net.eval()
    out_clear = net(dense_x, sparse_x)

    # Fit data to generate input & output ranges/shapes.
    orion.fit(net, [dense_x, sparse_x], batch_size=1)

    # Compile everything    
    input_level = orion.compile(net)

    # Encode and encrypt the input vector 
    dense_ctxt  = orion.encrypt(orion.encode(dense_x, input_level))
    sparse_ctxt = orion.encrypt(orion.encode(sparse_x, input_level))

    net.he()  # Switch to FHE mode

    # Run FHE inference
    print("\nStarting FHE inference", flush=True)
    start = time.time()
    out_ctxt = net(dense_ctxt, sparse_ctxt)
    end = time.time()

    # Get the FHE results and decrypt + decode.
    out_ptxt = out_ctxt.decrypt()
    out_fhe = out_ptxt.decode()

    # Compare the cleartext and FHE results.
    print()
    print(out_clear)
    print(out_fhe)

    dist = mae(out_clear, out_fhe)
    prec = math.inf if dist == 0 else -math.log2(dist)

    print(f"\nMAE: {dist:.4f}")
    print(f"Precision: {'∞' if prec == math.inf else f'{prec:.4f}'}")
    print(f"Runtime: {end-start:.4f} secs.\n")


if __name__ == "__main__":
    main()