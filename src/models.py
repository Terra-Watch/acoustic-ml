"""The two on-device students (channels all multiples of 16, no dense layers, ops the microcontroller runtime supports) and a MAC counter."""
import torch
import torch.nn as nn

from data import CLASSES


def conv_bn(cin, cout, k, s, groups=1, act=True):
    layers = [nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False), nn.BatchNorm2d(cout)]
    if act: layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class InvertedResidual(nn.Module):
    """1x1 expand, 3x3 depthwise, 1x1 project; residual when the shape is unchanged (the block family of the DCASE low-complexity baselines)."""
    def __init__(self, cin, cout, stride, expand):
        super().__init__(); mid = cin * expand
        self.body = nn.Sequential(conv_bn(cin, mid, 1, 1), conv_bn(mid, mid, 3, stride, groups=mid), conv_bn(mid, cout, 1, 1, act=False))
        self.res = stride == 1 and cin == cout

    def forward(self, x):
        y = self.body(x); return x + y if self.res else y


class StudentIR(nn.Module):
    """Inverted-residual student in the style of the DCASE CP-Mobile baseline (a simplified network, not the identical one)."""
    def __init__(self, width=1, n_classes=len(CLASSES)):
        super().__init__()
        cfg = [(16, 16, 1, 2), (16, 32, 2, 3), (32, 32, 1, 3), (32, 48, 2, 3), (48, 48, 1, 3), (48, 64, 2, 3), (64, 64, 1, 3)]
        blocks, c = [conv_bn(1, 16, 3, 2)], 16
        for cin, cout, s, e in cfg: blocks.append(InvertedResidual(cin * width, cout * width, s, e))
        self.features = nn.Sequential(*blocks)
        self.head = nn.Conv2d(64 * width, n_classes, 1)           # 1x1 conv on the pooled map = the classifier, no dense layer

    def forward(self, x):
        return self.head(self.features(x).mean((2, 3), keepdim=True)).flatten(1)


class DSCNN(nn.Module):
    """Plain depthwise-separable CNN (the control): stem, four depthwise 3x3 + pointwise blocks, pool, 1x1 classifier."""
    def __init__(self, n_classes=len(CLASSES)):
        super().__init__()
        ch = [32, 64, 64, 128, 128]; layers = [conv_bn(1, ch[0], 3, 2)]
        for i in range(1, len(ch)):
            layers += [conv_bn(ch[i - 1], ch[i - 1], 3, 2 if i in (2, 4) else 1, groups=ch[i - 1]), conv_bn(ch[i - 1], ch[i], 1, 1)]
        self.features = nn.Sequential(*layers); self.head = nn.Conv2d(ch[-1], n_classes, 1)

    def forward(self, x):
        return self.head(self.features(x).mean((2, 3), keepdim=True)).flatten(1)


MODELS = {"student_ir": StudentIR, "dscnn": DSCNN}


def count(model, shape=(1, 1, 40, 101)):
    """(parameters, multiply-accumulates) of one forward pass."""
    macs = [0]
    def hook(m, i, o): macs[0] += o.shape[2] * o.shape[3] * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    model.eval()
    with torch.no_grad(): model(torch.zeros(shape))
    for h in hs: h.remove()
    return sum(p.numel() for p in model.parameters()), macs[0]
