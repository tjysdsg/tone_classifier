import numpy as np
import os
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def collate_fn_pad(batch):
    transposed = list(zip(*batch))
    x = transposed[0]
    y = transposed[1]
    y = torch.from_numpy(np.asarray(y, dtype='int64'))
    x = pad_sequence(x, batch_first=True)
    return x, y


def collate_sequential_spectorgram(batch):
    transposed: list = list(zip(*batch))
    utts, xs, ys = transposed
    return utts, xs, ys


def collate_sequential_embedding(batch):
    transposed: list = list(zip(*batch))
    x: list = transposed[0]  # (batch_size, seq_len, embd_size)
    lengths = [len(e) for e in x]

    y = transposed[1]  # (batch_size, seq_len)
    x = pad_sequence(x, batch_first=True)
    y = pad_sequence(y, batch_first=True, padding_value=-100)  # -100 is ignored by NLLLoss

    return x, torch.as_tensor(y, dtype=torch.long), lengths


class SpectrogramDataset(Dataset):
    def __init__(self, wav_scp, utt2label):
        self.wav_scp = wav_scp
        self.utt2label = utt2label

    def __len__(self):
        return len(self.wav_scp)

    def __getitem__(self, idx):
        utt, filename = self.wav_scp[idx]
        signal = np.load(filename, allow_pickle=False)
        signal = np.moveaxis(signal, 0, 1)
        signal = torch.from_numpy(signal.astype('float32'))
        return signal, self.utt2label[utt]


class SequentialSpectrogramDataset(Dataset):
    def __init__(self, utts: list, utt2tones: dict):
        self.utts = utts
        self.utt2tones = utt2tones

        self.sequences = []
        for utt in self.utts:
            data = self.utt2tones[utt]
            self.sequences.append((utt, data))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        from feature_extraction import get_output_path
        utt, seq = self.sequences[idx]

        xs = []
        ys = []
        for tone, phone, start, dur in seq:
            path = get_output_path(utt, phone, start, f'feats/{tone}')
            x = np.load(path, allow_pickle=False)
            x = np.moveaxis(x, 0, 1)
            x = torch.as_tensor(x, dtype=torch.float32)  # sig_len * mels
            xs.append(x)
            ys.append(tone)

        # xs: (seq_len, sig_len, mels)
        # ys: (seq_len,)
        return utt, xs, ys


class SequentialEmbeddingDataset(Dataset):
    def __init__(self, utts: list, utt2tones: dict, embedding_dir='embeddings'):
        self.utts = utts
        self.utt2tones = utt2tones
        self.embedding_dir = embedding_dir

        self.sequences = []
        for utt in self.utts:
            data = self.utt2tones[utt]
            self.sequences.append((utt, data))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        utt, seq = self.sequences[idx]
        path = os.path.join(self.embedding_dir, f'{utt}.npy')
        embeddings = np.load(path, allow_pickle=False)
        xs = embeddings[:len(seq)]

        ys = []
        for tone, phone, start, dur in seq:
            ys.append(tone)

        # (seq_len, embd_size) and (seq_len,)
        return torch.as_tensor(xs, dtype=torch.float32), torch.as_tensor(ys, dtype=torch.long)
