import torch.nn as nn
import torch
from . import utils

__all__ = [
    'Modelv1',
    'Modelv2',
    'Modelv3',
    'select'
]


class Modelv1(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, input_size=32, output_dim=1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) *
                      (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid() if output_dim == 1 else nn.Softmax(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)

        return x


class Modelv2(nn.Module):
    # A smaller one
    def __init__(self, input_dim=1, input_size=32, output_dim=1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 30, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(30, 50, 4, 2, 1),
            nn.BatchNorm2d(50),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(50 * (self.input_size // 4) *
                      (self.input_size // 4), 100),
            nn.BatchNorm1d(100),
            nn.LeakyReLU(0.2),
            nn.Linear(100, self.output_dim),
            nn.Sigmoid() if output_dim == 1 else nn.Softmax(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 50 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)

        return x


class Modelv3(nn.Module):
    # A smaller one
    def __init__(self, input_dim=1, input_size=32, output_dim=1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 15, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(15, 40, 4, 2, 1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(40 * (self.input_size // 4) *
                      (self.input_size // 4), 50),
            nn.ReLU(),
            nn.Linear(50, self.output_dim),
            nn.Sigmoid() if output_dim == 1 else nn.Softmax(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 40 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)

        return x


class Ensemble(nn.Module):
    def __init__(self, v, n, input_dim=1, input_size=32, output_dim=1):
        assert v != 'ensemble'
        assert n >= 1
        super().__init__()
        for i in range(n):
            self.add_module(
                str(i),
                select['v%s' % v](
                    input_dim=input_dim,
                    input_size=input_size,
                    output_dim=output_dim)
            )

    def forward(self, x):
        preds = 0.
        for model in self._modules.values():
            preds += model(x)
        return preds / len(self._modules)


def add_coords(input_tensor, with_r=False):
    """
    input_tensor: (batch, c, x_dim, y_dim)
    """
    assert input_tensor.dim() == 4
    batch_size_tensor, _, x_dim, y_dim = input_tensor.shape

    xx_channel = (torch.arange(-1 + 1 / x_dim, 1. + 1 / x_dim, 2 / x_dim, device=input_tensor.device)
                  .view(1, 1, x_dim, 1)
                  .repeat(batch_size_tensor, 1, 1, y_dim))

    yy_channel = (torch.arange(-1 + 1 / y_dim, 1. + 1 / y_dim, 2 / y_dim, device=input_tensor.device)
                  .view(1, 1, 1, y_dim)
                  .repeat(batch_size_tensor, 1, x_dim, 1))
    channels = [xx_channel, yy_channel]
    if with_r:
        rr_channel = torch.sqrt(torch.pow(xx_channel, 2) + torch.pow(yy_channel, 2))
        channels.append(rr_channel)

    ret = torch.cat(channels + [input_tensor], dim=1)
    return ret


class CoordConv2d(nn.Conv2d):
    """CoordConv layer as in the paper."""

    def __init__(self, in_channels, *args, with_r=False, **kwargs):
        self.with_r = with_r
        super().__init__(in_channels + 2 + int(bool(with_r)),
                         *args, **kwargs)

    def forward(self, input_tensor):
        ret = add_coords(input_tensor, self.with_r)
        ret = super().forward(ret)
        return ret


class Modelv4(nn.Module):
    def __init__(self, input_dim=1, input_size=32, output_dim=1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.c = c = [16, 32]
        self.conv = nn.Sequential(
            CoordConv2d(self.input_dim, c[0], 5, 1, 2, bias=False),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
            CoordConv2d(c[0], c[1], 3, 1, 1, bias=False),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
        )
        self.fc0 = nn.Sequential(
            nn.Linear(c[1]*9, 20),
            nn.Tanh(),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(20+2, 1),
            nn.Sigmoid(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, self.c[-1]*9)
        x = self.fc0(x)
        vi = input.contiguous().view(x.shape[0], -1)
        x = torch.cat([x, vi.mean(-1)[:, None], vi.std(-1)[:, None]], -1)
        x = self.fc1(x)

        return x


select = dict(
    v1=Modelv1,
    v2=Modelv2,
    v3=Modelv3,
    v4=Modelv4,
    ensemble=Ensemble
)
