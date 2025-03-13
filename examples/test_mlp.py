import time
import math
import torch
import orion
import models
from orion.core.utils import (
    get_mnist_datasets,
    mae, 
    train_on_mnist
)

# Set seed for reproducibility
torch.manual_seed(42)

# Initialize model and data
scheme = orion.init_scheme("test_parameters.yaml")
batch_size = scheme.get_batch_size()
trainloader, testloader = get_mnist_datasets(data_dir="../data", batch_size=batch_size)
net = models.MLP()

# Train model (optional)
train_on_mnist(net, data_dir="../data", epochs=1, device="cuda")

# Get test batch for inference comparison
inp, _ = next(iter(testloader))

# Run cleartext inference
net.eval()
out_clear = net(inp)

# Prepare for FHE inference. 
# We'll determine the ranges with which to fit each polynomial activation
# function to by gathering these statistics from the training set and 
# applying a tolerance factor = margin.
orion.fit(net, trainloader, batch_size=128, margin=2)
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