import time
import math
import torch
import orion
import orion.nn as on
from orion.core.utils import get_mnist_datasets, mae 

orion.set_log_level('DEBUG')
torch.manual_seed(42)


class Layer(on.Module):
    def __init__(self):
        super().__init__()
        #self.set_depth(0)
        self.trace_internal_ops(True)
    
    def forward(self, x):
        return x, 5.5*x


class Concat(on.Module):
    def __init__(self):
        super().__init__()
        #self.set_depth(0)
        self.trace_internal_ops(True)
    
    def forward(self, x, y, z):
        return x + y + z


class DAG_TEST(on.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.flatten = on.Flatten()
        self.splitter = on.Linear(784, 128)

        # self.fc1 = on.Linear(128, 128)
        # self.fc2 = on.Linear(128, 128)
        # self.fc3 = on.Linear(128, 128)

        self.layer1 = Layer()
        self.layer2 = Layer()
        self.layer3 = Concat()

    def forward(self, x): 
        x = self.flatten(x)
        x = self.splitter(x)

        # a = self.fc1(x)
        # b = self.fc2(x)
        # c = self.fc3(x)

        a, b = self.layer1(x)
        c, d = self.layer2(a)
        e = self.layer3(b, c, d)

        return e


def main():
    scheme = orion.init_scheme("../configs/mlp.yml")

    batch_size = scheme.params.get_batch_size()
    trainloader, _ = get_mnist_datasets(data_dir="../data", batch_size=batch_size)
    net = DAG_TEST()

    inp, _ = next(iter(trainloader))

    # Run cleartext inference
    net.eval()
    out_clear = net(inp)

    # Prepare for FHE inference. 
    orion.fit(net, inp)
    input_level = orion.compile(net)

    # Encode and encrypt the input vector 
    vec_ctxt = orion.encrypt(orion.encode(inp, input_level))
    net.he()  # Switch to FHE mode

    out_ctxt = net(vec_ctxt)

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
