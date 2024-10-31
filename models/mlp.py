import orion.nn as on

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
    

if __name__ == "__main__":
    import torch
    from torchsummary import summary
    from fvcore.nn import FlopCountAnalysis

    net = MLP()
    net.eval()

    x = torch.randn(1,1,28,28)
    total_flops = FlopCountAnalysis(net, x).total()

    summary(net, (1,28,28), depth=10)
    print("Total flops: ", total_flops)
