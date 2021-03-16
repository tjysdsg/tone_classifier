import torch.nn as nn
from train.modules.front_resnet import ResNet34
from train.modules.pooling import StatsPool, ScaleDotProductAttention


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
