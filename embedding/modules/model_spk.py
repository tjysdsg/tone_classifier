import numpy as np
import torch.nn as nn
from embedding.modules.front_resnet import ResNet34
from embedding.modules.pooling import StatsPool, ScaleDotProductAttention


class ResNet34StatsPool(nn.Module):

    def __init__(self, in_planes, embedding_size, dropout=0.5, **kwargs):
        super(ResNet34StatsPool, self).__init__()
        self.front = ResNet34(in_planes, **kwargs)
        self.pool = StatsPool()
        self.bottleneck = nn.Linear(in_planes * 8 * 2, embedding_size)
        self.drop = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        x = self.front(x.unsqueeze(dim=1))
        x = self.pool(x)
        x = self.bottleneck(x)
        if self.drop:
            x = self.drop(x)
        return x


class FTDNNStatsPool(nn.Module):

    def __init__(self, in_planes, embedding_size, total_step=200, dropout=0.5, factorize_step_size=4):

        super(FTDNNStatsPool, self).__init__()
        self.front = FTDNN(in_planes)
        self.pool = StatsPool()
        self.bottleneck = nn.Linear(4096, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)
        self.nl = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout) if dropout else None

        self.step = 0
        self.drop_schedule = np.interp(np.linspace(0, 1, total_step), [0, 0.5, 1], [0, 0.5, 0])
        self.factorize_step_size = factorize_step_size

    def forward(self, x):
        if self.training:
            self.front.set_dropout_alpha(self.drop_schedule[self.step])
            if self.step % self.factorize_step_size == 1:
                self.front.step_ftdnn_layers()
            self.step += self.step

        x = self.front(x)
        x = self.pool(x)
        x = self.bottleneck(x)
        x = self.nl(x)
        x = self.bn(x)
        if self.drop:
            x = self.drop(x)
        return x


class ResNet34SDPAttPool(nn.Module):

    def __init__(self, in_planes, embedding_size, dropout=0.5, **kwargs):
        super(ResNet34SDPAttPool, self).__init__()
        self.front = ResNet34(in_planes, **kwargs)
        self.pool = ScaleDotProductAttention(in_planes * 8)
        self.bottleneck = nn.Linear(in_planes * 8, embedding_size)
        self.drop = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        x = self.front(x.unsqueeze(dim=1))  # batch x channel x freq x time
        x = x.mean(dim=2).transpose(1, 2)  # batch x time x feat_dim
        x = self.pool(x)
        x = self.bottleneck(x)
        if self.drop:
            x = self.drop(x)
        return x
