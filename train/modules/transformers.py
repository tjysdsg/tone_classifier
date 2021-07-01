import torch
import math
import torch.nn as nn
from train.utils import get_padding_mask


class PositionalEncoding(nn.Module):
    """
    Input size:
    (seq_len, batch_size, feature_size)
    """

    def __init__(self, d_model, max_len: int, dropout=0.1):
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
        x = x + self.pe[:x.size(0)]
        # print(x[:, 0, :])
        return self.dropout(x)


class TransEncoder(nn.Module):
    """
    Batch-first Transformer encoder
    """

    def __init__(self, num_classes: int, num_layers=6, embedding_size=128, nhead=16, max_len=100):
        super().__init__()
        self.pos_enc = PositionalEncoding(embedding_size, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=nhead)
        encoder_norm = nn.LayerNorm(embedding_size)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=encoder_norm
        )

        self.fc = nn.Linear(embedding_size, num_classes)

    def forward(self, x, lengths):
        # build src_key_padding_mask
        x = x.cuda()
        padding_mask = get_padding_mask(x, lengths).cuda()

        # convert to seq_len * batch_size * hidden
        x = x.transpose(0, 1)
        x = self.pos_enc(x)
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        # convert to batch_size * seq_len * hidden
        x = x.transpose(0, 1)
        x = self.fc(x)
        return x, padding_mask


def test():
    net = TransEncoder(5)
    a = [torch.zeros(20, 128), torch.zeros(15, 128), torch.zeros(25, 128)]  # batch * seq * feat
    lengths = [20, 15, 25]
    a = torch.nn.utils.rnn.pad_sequence(a, batch_first=True)
    a = torch.nn.utils.rnn.pack_padded_sequence(a, lengths, batch_first=True, enforce_sorted=False)
    b, padding_mask = net(a)
    print(b.shape)
    print(padding_mask)


if __name__ == '__main__':
    test()
