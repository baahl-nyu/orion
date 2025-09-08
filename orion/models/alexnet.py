import torch.nn as nn
import orion.nn as on


class ConvBlock(on.Module):
    def __init__(self, Ci, Co, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Sequential(
            on.Conv2d(Ci, Co, kernel_size, stride, padding, bias=False),
            on.BatchNorm2d(Co),
            on.SiLU(degree=127))
    
    def forward(self, x):
        return self.conv(x)
    

class LinearBlock(on.Module):
    def __init__(self, ni, no):
        super().__init__()
        self.linear = nn.Sequential(
            on.Linear(ni, no),
            on.BatchNorm1d(no),
            on.SiLU(degree=127))
        
    def forward(self, x):
        return self.linear(x)


class AlexNet(on.Module):
    cfg = [64, 'M', 192, 'M', 384, 256, 256, 'A']

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = self._make_layers()
        self.flatten = on.Flatten()
        self.classifier = nn.Sequential(
            LinearBlock(1024, 4096),
            LinearBlock(4096, 4096),
            on.Linear(4096, num_classes))

    def _make_layers(self):
        layers = []
        in_channels = 3
        for x in self.cfg:
            if x == 'M':
                layers += [on.AvgPool2d(kernel_size=2, stride=2)]
            elif x == 'A':
                layers += [on.AdaptiveAvgPool2d((2, 2))]
            else:
                layers += [ConvBlock(in_channels, x, kernel_size=3, stride=1, padding=1)]
                in_channels = x
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    

if __name__ == "__main__":
    import torch
    from torchsummary import summary
    from fvcore.nn import FlopCountAnalysis

    net = AlexNet()
    net.eval()

    x = torch.randn(1,3,32,32)
    total_flops = FlopCountAnalysis(net, x).total()

    summary(net, (3,32,32), depth=10)
    print("Total flops: ", total_flops)
