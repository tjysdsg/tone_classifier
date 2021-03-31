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
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = ScaleDotProductAttention(hidden_size)

    def forward(self, x):
        """
        Input size: (batch, time, feat_size)
        """
        att_weights = self.attention(x)

        weighted_x = torch.mul(x, att_weights)
        mean = torch.mean(weighted_x, dim=1)

        weighted_x_sq = torch.mul(x, weighted_x)
        variance = torch.sum(weighted_x_sq, dim=1) - torch.mul(mean, mean)

        ret = torch.cat((mean, variance), 1)
        return ret


class ScaleDotProductAttention(nn.Module):

    def __init__(self, embed_dim):
        super(ScaleDotProductAttention, self).__init__()
        self.scaling = float(embed_dim) ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        q = self.q_proj(x) * self.scaling
        k = self.k_proj(x)
        attn_output_weights = F.softmax(torch.bmm(q, k.transpose(1, 2)), dim=-1)
        return torch.bmm(attn_output_weights, x)
