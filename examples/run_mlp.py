import time
import math
import torch
import orion
import orion.nn as on
from orion.core.utils import (
    get_mnist_datasets,
    mae, 
)

orion.set_log_level('DEBUG')

class MLP(on.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.flatten = on.Flatten()
        
        self.fc1 = on.Linear(784, 128)
        self.bn1 = on.BatchNorm1d(128)
        self.act1 = on.Quad()
        
        self.fc2 = on.Linear(128, 128)
        self.bn2 = on.BatchNorm1d(128)
        self.act2 = on.Quad() 
        
        self.fc3 = on.Linear(128, num_classes)

    def forward(self, x): 
        x = self.flatten(x)
        x = self.act1(self.bn1(self.fc1(x)))
        x = self.act2(self.bn2(self.fc2(x)))
        return self.fc3(x)

# Set seed for reproducibility
torch.manual_seed(42)

# Initialize the Orion scheme, model, and data
scheme = orion.init_scheme("../configs/mlp.yml")
trainloader, testloader = get_mnist_datasets(data_dir="../data")
net = MLP()

#inp = torch.randn(1, 784)
inp, _ = next(iter(trainloader))
inp = inp[0].unsqueeze(0)

# Run cleartext inference
net.eval()
out_clear = net(inp)


# Prepare for FHE inference. 
orion.fit(net, inp)
input_level = orion.compile(net)

# Encode and encrypt the input vector 
vec_ptxt = orion.encode(inp, input_level)
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
print(out_clear)
print(out_fhe)

dist = mae(out_clear, out_fhe)
print(f"\nMAE: {dist:.4f}")
print(f"Precision: {-math.log2(dist):.4f}")
print(f"Runtime: {end-start:.4f} secs.\n")