import torch.nn as nn
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


select = dict(
    v1=Modelv1,
    v2=Modelv2,
    v3=Modelv3,
    v4=Modelv4
)
