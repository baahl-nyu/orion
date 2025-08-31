import time
import math
import torch
import orion
import orion.nn as on

orion.set_log_level('DEBUG')

class TensorContraction(on.Module):
    def __init__(self):
        super().__init__()
        self.contraction1 = on.Einsum("ij->")

    def forward(self, x): 
        return self.contraction1(x)

# Set seed for reproducibility
torch.manual_seed(42)

# Initialize the Orion scheme, model, and data
scheme = orion.init_scheme("../configs/resnet.yml")
net = TensorContraction()
net.set_scheme(scheme)


inp1 = torch.arange(1, 2*3 + 1.).view(2, 3)
# inp2 = torch.arange(1, 3*4 + 1.).view(3, 4)

# Run cleartext inference
net.eval()
out_clear = net(inp1)

# Prepare for FHE inference. 
orion.fit(net, [inp1])
input_level = orion.compile(net)

# Encode and encrypt the input vector 
vec_ptxt = orion.encode(inp1, input_level)
vec_ctxt = orion.encrypt(vec_ptxt)
# vec_ptxt2 = orion.encode(inp2, input_level)
# vec_ctxt2 = orion.encrypt(vec_ptxt2)
net.he()  # Switch to FHE mode

# Run FHE inference
print("\nStarting FHE inference", flush=True)
start = time.time()
out_ctxt = net(vec_ctxt)
end = time.time()

# Get the FHE results and decrypt + decode.
out_ptxt = out_ctxt.decrypt()
out_fhe = out_ptxt.decode()

# Compare the cleartext and FHE results.
print()
print('input1:', inp1)
# print('input2:', inp2)
print('clear_out:', out_clear)
print('fhe_out:', out_fhe)
print('L2 norm:', torch.norm(out_clear - out_fhe))