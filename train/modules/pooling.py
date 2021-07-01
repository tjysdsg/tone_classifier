import torch
import torch.nn as nn
import torch.nn.functional as F


class StatsPool(nn.Module):

    def __init__(self):
        super(StatsPool, self).__init__()

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], -1)
        out = torch.cat([x.mean(dim=2), x.std(dim=2)], dim=1)
        return out


class AvgPool(nn.Module):

    def __init__(self):
        super(AvgPool, self).__init__()

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], -1)
        return x.mean(dim=2)


class AttStatsPool(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.attention = ScaledDotProductAttention(hidden_size)

    def forward(self, x):
        """
        Input size: (batch, time, feat_size)
        """
        weighted = self.attention(x)

        mean = torch.mean(weighted, dim=1)
        weighted_sq = torch.mul(x, weighted)
        variance = torch.sum(weighted_sq, dim=1) - torch.mul(mean, mean)

        ret = torch.cat((mean, variance), dim=1)
        return ret


class ScaledDotProductAttention(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.scaling = float(input_size) ** -0.5
        self.q = nn.Linear(input_size, input_size)
        self.k = nn.Linear(input_size, input_size)
        self.v = nn.Linear(input_size, input_size)

    def forward(self, x):
        """
        Input shape: (B, T, D)
        Output shape: (B, T, D)
        """
        q = self.q(x) * self.scaling  # (B, T, D)
        k = self.k(x)  # (B, T, D)
        v = self.v(x)  # (B, T, D)
        attn_weights = F.softmax(torch.bmm(q, k.transpose(1, 2)), dim=-1)  # (B, T, T)
        return torch.bmm(attn_weights, v)  # (B, T, D)
