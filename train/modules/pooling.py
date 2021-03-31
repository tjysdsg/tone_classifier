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
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.attention = ClassicAttention(input_size, hidden_size)

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


class ClassicAttention(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, attn_dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.lin_proj = nn.Linear(input_dim, embed_dim)
        self.v = torch.nn.Parameter(torch.randn(embed_dim))

    def forward(self, x):
        lin_out = self.lin_proj(x)
        v_view = self.v.unsqueeze(0).expand(lin_out.size(0), len(self.v)).unsqueeze(2)
        attention_weights = F.tanh(lin_out.bmm(v_view).squeeze())
        attention_weights_normalized = F.softmax(attention_weights, 1)
        return attention_weights_normalized
