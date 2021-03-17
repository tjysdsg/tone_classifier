import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset


def collate_fn_pad(batch):
    transposed = list(zip(*batch))
    x = transposed[0]
    y = transposed[1]
    y = torch.from_numpy(np.asarray(y, dtype='int64'))
    x = pad_sequence(x, batch_first=True)
    return x, y


def collate_fn_pack_pad(batch):
    transposed = list(zip(*batch))
    x = transposed[0]
    x_lens = [len(e) for e in x]

    y = transposed[1]
    y = torch.from_numpy(np.asarray(y, dtype='int64'))
    x = pad_sequence(x, batch_first=True)

    packed = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
    return packed, y


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
        return len(self.utts)

    def __getitem__(self, idx):
        from feature_extraction import get_output_path
        utt = self.utts[idx]
        print('utt', utt)
        data = self.utt2tones[utt]
        y = []
        x = []
        for tone, phone, start, dur in data:
            path = get_output_path(utt, phone, start, f'feats/{tone}')
            signal = np.load(path, allow_pickle=False)
            signal = np.moveaxis(signal, 0, 1)
            signal = torch.from_numpy(signal.astype('float32'))
            embd = self.embd_model(signal)
            print('Embd shape', embd.shape)
            y.append(tone)
            x.append(embd)
        y = torch.from_numpy(np.asarray(y, dtype='int64'))
        x = torch.stack(x, dim=0)
        return x, y
