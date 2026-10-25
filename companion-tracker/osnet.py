"""OSNet person re-identification network (inference only).

Adapted from deep-person-reid / torchreid by Kaiyang Zhou (MIT License,
https://github.com/KaiyangZhou/deep-person-reid). Training code and ImageNet download removed.
Reference: Zhou et al., "Omni-Scale Feature Learning for Person Re-Identification", ICCV 2019.
"""
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F


class ConvLayer(nn.Module):
    def __init__(self, cin, cout, k, stride=1, padding=0, groups=1, IN=False):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=stride, padding=padding, bias=False, groups=groups)
        self.bn = nn.InstanceNorm2d(cout, affine=True) if IN else nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    def __init__(self, cin, cout, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 1, stride=stride, padding=0, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False, groups=cout)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class ChannelGate(nn.Module):
    def __init__(self, cin, reduction=16):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(cin, cin // reduction, kernel_size=1, bias=True, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(cin // reduction, cin, kernel_size=1, bias=True, padding=0)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x):
        g = self.gate_activation(self.fc2(self.relu(self.fc1(self.global_avgpool(x)))))
        return x * g


class OSBlock(nn.Module):
    def __init__(self, cin, cout, IN=False, bottleneck_reduction=4):
        super().__init__()
        mid = cout // bottleneck_reduction
        self.conv1 = Conv1x1(cin, mid)
        self.conv2a = LightConv3x3(mid, mid)
        self.conv2b = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid))
        self.conv2c = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(3)])
        self.conv2d = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(4)])
        self.gate = ChannelGate(mid)
        self.conv3 = Conv1x1Linear(mid, cout)
        self.downsample = Conv1x1Linear(cin, cout) if cin != cout else None
        self.IN = nn.InstanceNorm2d(cout, affine=True) if IN else None

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = (self.gate(self.conv2a(x1)) + self.gate(self.conv2b(x1))
              + self.gate(self.conv2c(x1)) + self.gate(self.conv2d(x1)))
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        if self.IN is not None:
            out = self.IN(out)
        return F.relu(out)


class OSNet(nn.Module):
    def __init__(self, channels, layers=(2, 2, 2), feature_dim=512, IN=False):
        super().__init__()
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=IN)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(layers[0], channels[0], channels[1], True, IN)
        self.conv3 = self._make_layer(layers[1], channels[1], channels[2], True)
        self.conv4 = self._make_layer(layers[2], channels[2], channels[3], False)
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channels[3], feature_dim), nn.BatchNorm1d(feature_dim),
                                nn.ReLU(inplace=True))

    @staticmethod
    def _make_layer(n, cin, cout, reduce, IN=False):
        layers = [OSBlock(cin, cout, IN=IN)] + [OSBlock(cout, cout, IN=IN) for _ in range(1, n)]
        if reduce:
            layers.append(nn.Sequential(Conv1x1(cout, cout), nn.AvgPool2d(2, stride=2)))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv5(self.conv4(self.conv3(self.conv2(self.maxpool(self.conv1(x))))))
        return self.fc(self.global_avgpool(x).flatten(1))


ARCHS = {
    "osnet_x1_0": dict(channels=(64, 256, 384, 512)),
    "osnet_x0_75": dict(channels=(48, 192, 288, 384)),
    "osnet_x0_5": dict(channels=(32, 128, 192, 256)),
    "osnet_x0_25": dict(channels=(16, 64, 96, 128)),
    "osnet_ibn_x1_0": dict(channels=(64, 256, 384, 512), IN=True),
}


def arch_from_filename(path):
    name = str(path).replace("\\", "/").rsplit("/", 1)[-1]
    for key in sorted(ARCHS, key=len, reverse=True):   # longest first: osnet_ibn_x1_0 before osnet_x1_0
        if name.startswith(key):
            return key
    raise ValueError(f"Cannot tell the OSNet size from '{name}'. Name it like osnet_x1_0_msmt17.pt")


def load_osnet(path, arch=None):
    """Build the network and load a torchreid / boxmot checkpoint. Fails loudly on a bad file."""
    arch = arch or arch_from_filename(path)
    model = OSNet(**ARCHS[arch])
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)   # older pickled checkpoints
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    own = model.state_dict()
    clean = OrderedDict()
    for k, v in sd.items():
        k = k[7:] if k.startswith("module.") else k
        if k in own and own[k].shape == v.shape:
            clean[k] = v
    missing = [k for k in own if k not in clean and not k.endswith("num_batches_tracked")]
    if missing:
        raise RuntimeError(f"{path}: {len(missing)} weights missing (e.g. {missing[:3]}). "
                           f"Wrong file or wrong architecture '{arch}'.")
    own.update(clean)
    model.load_state_dict(own)
    return model.eval()
