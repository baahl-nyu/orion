import time
import math
import torch
import orion
import orion.nn as on

orion.set_log_level('DEBUG')

class Attention(on.Module):
    def __init__(self, hidden_dim, head_dim):
        super().__init__()
        self.q = on.Linear(hidden_dim, head_dim, bias=False)
        self.k = on.Linear(hidden_dim, head_dim, bias=False)
        self.v = on.Linear(hidden_dim, head_dim, bias=False)
        self.attention_score = on.Einsum("th,Th->tT")
        self.linear_comb = on.Einsum("tT,Th->th")

    def forward(self, x): 
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        qk = self.attention_score(q, k)
        return self.linear_comb(qk, v)

# Set seed for reproducibility
torch.manual_seed(42)

num_tokens = 4
hidden_dim = 16
num_heads = 2
head_dim = hidden_dim // num_heads
inp1 = torch.randn(num_tokens, hidden_dim)

# Initialize the Orion scheme, model, and data
scheme = orion.init_scheme("../configs/resnet.yml")
net = Attention(hidden_dim, head_dim)
net.set_scheme(scheme)



# Run cleartext inference
net.eval()
out_clear = net(inp1)


# Prepare for FHE inference. 
orion.fit(net, [inp1])
input_level = orion.compile(net)

# Encode and encrypt the input vector 
vec_ptxt = orion.encode(inp1, input_level)
vec_ctxt = orion.encrypt(vec_ptxt)
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
print('clear_out:', out_clear)
print('fhe_out:', out_fhe)
print('L2 norm:', torch.norm(out_clear - out_fhe))