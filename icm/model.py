

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):

    def __init__(self, in_dim, out_dim, d_model=128, n_layers=2, discrete=False):
        super(FeedForward, self).__init__()
        self.discrete = discrete

        self.in_layer = nn.Linear(in_dim, d_model)
        self.layers = nn.ModuleList([
            nn.Linear(d_model, d_model)
            for _ in range(n_layers)
        ])
        self.out_layer = nn.Linear(d_model, out_dim)

    def forward(self, x):
        x = F.relu(self.in_layer(x))
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.out_layer(x)

        if self.discrete:
            x = F.softmax(x, dim=-1)

        return x


class ConvNet(nn.Module):

    def __init__(self, channels, out_dim, discrete=False):
        super(ConvNet, self).__init__()
        self.discrete = discrete

        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(3456, 512)
        self.out = nn.Linear(512, out_dim)

    def forward(self, x):
        # channels last to channels first
        x = x.permute(0, 3, 1, 2)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)

        x = F.relu(self.fc(x))
        x = self.out(x)

        if self.discrete:
            x = F.softmax(x, dim=-1)

        return x

