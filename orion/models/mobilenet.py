import torch.nn as nn 
import orion.nn as on

class Block(on.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.conv1 = on.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, 
                               padding=1, groups=in_planes, bias=False)
        self.bn1 = on.BatchNorm2d(in_planes)
        self.act1 = on.SiLU(degree=127)

        self.conv2 = on.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                                padding=0, bias=False)
        self.bn2 = on.BatchNorm2d(out_planes)
        self.act2 = on.SiLU(degree=127)

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        return out


class MobileNet(on.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=200):
        super().__init__()
        self.conv1 = on.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = on.BatchNorm2d(32)
        self.act = on.SiLU(degree=127)

        self.layers = self._make_layers(in_planes=32)
        self.avgpool = on.AdaptiveAvgPool2d(output_size=(1,1))
        self.flatten = on.Flatten()
        self.linear = on.Linear(1024, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.flatten(self.avgpool(out))
        return self.linear(out)

if __name__ == "__main__":
    import torch
    from torchsummary import summary
    from fvcore.nn import FlopCountAnalysis

    net = MobileNet()
    net.eval()

    x = torch.randn(1,3,32,32)
    total_flops = FlopCountAnalysis(net, x).total()

    summary(net, (3,32,32), depth=10)
    print("Total flops: ", total_flops)