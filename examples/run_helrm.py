import time
import torch
import orion
from orion.models.helrm import HELRM

# Set seed for reproducibility
torch.manual_seed(42)

# Initialize the Orion scheme, model, and data
scheme = orion.init_scheme("./configs/resnet.yml")
dense_size = 4
hidden_dim = 2
net = HELRM(num_dense=4, vocab_sizes=[4, 3, 2], hidden_dim=2)

inp = torch.tensor(
    [0.1753, -0.9315, -1.5055, -0.6610, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]
)
inp = inp[None, :]
# Run cleartext inference

print("running in the clear")
net.eval()
net.load_state_dict(torch.load("./model/helrm.pth"))
out_clear = net(inp)
print("out_clear", out_clear)

orion.fit(net, inp)
input_level = orion.compile(net)

# # Encode and encrypt the input vector
vec_ptxt = orion.encode(inp, input_level)
vec_ctxt = orion.encrypt(vec_ptxt)
net.he()  # Switch to FHE mode

# # Run FHE inference
print("\nStarting FHE inference", flush=True)
start = time.time()
out_ctxt = net(vec_ctxt)
end = time.time()

# # Get the FHE results and decrypt + decode.
out_ptxt = out_ctxt.decrypt()
out_fhe = out_ptxt.decode()

# # Compare the cleartext and FHE results.
print(out_clear)
print(out_fhe)
print("time: ", end - start)
