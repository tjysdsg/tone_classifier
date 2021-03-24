import torch
import torch.nn as nn
from torch.functional import F
from train.modules.front_resnet import ResNet34
from train.modules.tdnn import TDNN
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


class _StatsPool(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = torch.cat([x.mean(dim=1), x.std(dim=1)], dim=1)
        return out


class TDNNStatsPool(nn.Module):
    def __init__(self, input_size=64, embedding_size=128):
        super().__init__()
        self.frame1 = TDNN(input_dim=input_size, output_dim=512, context_size=5, dilation=1)
        self.frame2 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=2)
        self.frame3 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=3)
        self.frame4 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1)
        self.frame5 = TDNN(input_dim=512, output_dim=1500, context_size=1, dilation=1)
        self.pool = _StatsPool()
        self.segment6 = nn.Linear(3000, embedding_size)

    def forward(self, x):
        """
        Input to frame1 is of shape (batch_size, T, 24)
        Output of frame5 is (batch_size, T - 14, 1500)
        Output of pool is (batch_size, 3000)
        """
        x = self.frame1(x)
        x = self.frame2(x)
        x = self.frame3(x)
        x = self.frame4(x)
        x = self.frame5(x)
        x = self.pool(x)
        x = F.relu(x)
        x = self.segment6(x)
        x = F.relu(x)
        return x
