import time
import math
import torch
import orion
import orion.nn as on
from orion.core.utils import get_mnist_datasets, mae 

orion.set_log_level('DEBUG')
torch.manual_seed(42)


class Layer1(on.Module):
    def __init__(self):
        super().__init__()
        self.set_depth(0)
        self.trace_internal_ops(False)
    
    def forward(self, a, b, c):
        return a + b + c



class DAG_TEST(on.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc1 = on.Linear(784, 128)
        self.fc2 = on.Linear(784, 128)
        self.fc3 = on.Linear(784, 128)
        self.fc4 = on.Linear(784, 128)

        self.layer1 = Layer1()

    def forward(self, x, y, z): 
        a = self.fc1(x)
        b = self.fc2(y)
        c = self.fc3(z)

        return self.layer1(a, b, c)


def main():
    scheme = orion.init_scheme("../configs/mlp.yml")

    batch_size = scheme.params.get_batch_size()
    trainloader, _ = get_mnist_datasets(data_dir="../data", batch_size=batch_size)
    net = DAG_TEST()

    inp1 = torch.randn(1, 784)
    inp2 = torch.randn(1, 784)
    inp3 = torch.randn(1, 784)

    # Run cleartext inference
    net.eval()
    out_clear = net(inp1, inp2, inp3)

    # Prepare for FHE inference. 
    orion.fit(net, [inp1, inp2, inp3])
    input_level = orion.compile(net)

    # Encode and encrypt the input vector 
    vec1_ctxt = orion.encrypt(orion.encode(inp1, input_level))
    vec2_ctxt = orion.encrypt(orion.encode(inp2, input_level))
    vec3_ctxt = orion.encrypt(orion.encode(inp3, input_level))
    net.he()  # Switch to FHE mode

    out_ctxt = net(vec1_ctxt, vec2_ctxt, vec3_ctxt)

    # Get the FHE results and decrypt + decode.
    out_fhe = out_ctxt.decrypt().decode()

    # Compare the cleartext and FHE results.
    print()
    print(out_clear)
    print(out_fhe)

    dist = mae(out_clear, out_fhe)
    print(f"\nMAE: {dist:.4f}")
    print(f"Precision: {-math.log2(dist):.4f}")


if __name__ == "__main__":
    main()
