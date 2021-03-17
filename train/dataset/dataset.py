import numpy as np
import torch
from torch.utils.data import Dataset


def collate_fn_pad(batch):
    transposed = list(zip(*batch))
    x = transposed[0]
    y = transposed[1]
    y = torch.from_numpy(np.asarray(y, dtype='int64'))
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
    return x, y


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


class EmbeddingDataset(Dataset):
    def __init__(self, utts: list, utt2tones: dict, embd_model):
        self.utts = utts
        self.utt2tones = utt2tones
        self.embd_model = embd_model

    def __len__(self):
        return len(self.utt2tones)

    def __getitem__(self, idx):
        from feature_extraction import get_output_path
        utt = self.utts[idx]
        data = self.utt2tones[utt]
        y = []
        x = []
        for tone, phone, start, dur in data:
            path = get_output_path(utt, phone, start, f'feats/{tone}')
            signal = np.load(path, allow_pickle=False)
            signal = np.moveaxis(signal, 0, 1)
            signal = torch.from_numpy(signal.astype('float32'))
            embd = self.embd_model(signal)
            y.append(tone)
            x.append(embd)
        y = torch.from_numpy(np.asarray(y, dtype='int64'))
        x = torch.from_numpy(np.asarray(x, dtype='float32'))
        return x, y
