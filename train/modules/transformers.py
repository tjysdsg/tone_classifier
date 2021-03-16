import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransEncoder(nn.Module):
    def __init__(self, num_classes: int, num_layers=6, embedding_size=128, nhead=8):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=nhead)
        self.pos_enc = PositionalEncoding(embedding_size)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        x = x.transpose(0, 1)
        x = self.pos_enc(x)
        x = self.transformer_encoder(x)
        x = self.fc(x)
        x = x.transpose(0, 1)
        x = F.softmax(x, -1)
        return x


def test():
    net = TransEncoder(4)
    a = torch.rand(32, 20, 128)  # batch * seq * feat
    b = net(a)
    print(b.shape)


if __name__ == '__main__':
    test()
