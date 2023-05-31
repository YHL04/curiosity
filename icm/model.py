

import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(FeedForward, self).__init__()

        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)

        return x
