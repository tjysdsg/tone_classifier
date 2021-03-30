import torch
from typing import List
import torch.nn as nn
import torch.nn.functional as F
from train.modules.resnet import ResNet34
from train.modules.tdnn import TDNN
from train.modules.pooling import StatsPool, ScaleDotProductAttention
from train.dataset.dataset import SpectroFeat
from torch.nn.utils.rnn import pad_sequence
from train.config import N_PHONES, SPEAKER_EMBEDDING_SIZE

__all__ = [
    'ResNet34SDPAttPool', 'ResNet34StatsPool', 'TDNNStatsPool', 'BLSTMStatsPool', 'EmbeddingModel', 'ContextualModel'
]


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


class BLSTMStatsPool(nn.Module):
    def __init__(self, input_size=64, embedding_size=128):
        super().__init__()
        hidden_size = 256
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True, dropout=0.3,
            bidirectional=True
        )
        self.pool = _StatsPool()
        self.embedding = nn.Linear(2 * 2 * hidden_size, embedding_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.pool(x)
        x = F.relu(x)
        x = self.embedding(x)
        x = F.relu(x)
        return x


class EmbeddingModel(nn.Module):
    def __init__(
            self, model: nn.Module, embedding_size: int, num_classes: int, hidden_size=128, include_segment_feats=False,
            include_context=False, include_spk=False
    ):
        super().__init__()
        self.include_segment_feats = include_segment_feats
        self.include_context = include_context
        self.include_spk = include_spk

        self.model1 = model

        seg_feat_size = 0
        if self.include_segment_feats:
            seg_feat_size += 1 + N_PHONES

        if self.include_spk:
            seg_feat_size += SPEAKER_EMBEDDING_SIZE

        if seg_feat_size > 0:
            if self.include_context:
                seg_feat_size *= 3
            self.model2 = nn.Linear(seg_feat_size, hidden_size)

        if include_segment_feats:
            self.classifier = nn.Linear(embedding_size + hidden_size, num_classes).cuda()
        else:
            self.classifier = nn.Linear(embedding_size, num_classes).cuda()

    def forward(self, x, durs=None, onehots=None, spk_embd=None):
        x = self.model1(x)  # (batch, embedding_size)
        x = F.relu(x)

        extra_feats = []
        if self.include_segment_feats:
            assert durs is not None and onehots is not None
            extra_feats.append(durs)
            extra_feats.append(onehots)

        if self.include_spk:
            assert spk_embd is not None
            extra_feats.append(spk_embd)

        if len(extra_feats) > 0:
            x1 = torch.hstack(extra_feats)
            x1 = torch.flatten(x1, 1)
            x1 = self.model2(x1)
            x1 = F.relu(x1)
            x = torch.hstack([x, x1])

        out = self.classifier(x)
        return out


class ContextualModel(nn.Module):
    def __init__(self, embd_model: nn.Module, model: nn.Module, include_dur=False):
        super().__init__()
        self.embd_model = embd_model
        self.model = model
        self.include_dur = include_dur

    def forward(self, packed):
        """
        X shape: (utts, seq_len, time, mels)
        """
        self.embd_model.eval()

        items: List = list(zip(*packed))
        feats: List[SpectroFeat] = items[0]  # (utts, seq_len', sig_len', mels)
        lengths = items[1]

        embds = []
        for feat in feats:
            x = feat.spectro
            x = x.type(torch.float32).cuda()  # (seq_len, sig_len, mels)

            embd = self.embd_model(x)

            if self.include_dur:
                sig_lens = torch.as_tensor(feat.lengths, dtype=torch.float32)
                embd = torch.stack([embd, sig_lens])

            embds.append(embd)  # embds: (utt, seq_len', embd_size)

        feat = pad_sequence(embds, batch_first=True)
        return self.model(feat, lengths)
