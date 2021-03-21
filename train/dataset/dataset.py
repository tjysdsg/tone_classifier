import numpy as np
import librosa
import random
import os
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from train.config import WAV_DIR, WAV_CACHE_DIR


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
    """
    Load extracted spectrogram
    """

    def __init__(self, wav_scp, utt2label):
        self.wav_scp = wav_scp
        self.utt2label = utt2label

    def __len__(self):
        return len(self.wav_scp)

    def __getitem__(self, idx):
        utt, filename = self.wav_scp[idx]

        # randomly choose data augmentation
        suffixes = ['', 'noise', 'sp09', 'sp11']
        suffix = suffixes[random.randint(0, len(suffixes) - 1)]
        if suffix != '':
            filename, ext = os.path.splitext(filename)
            filename = f'{filename}_{suffix}' + ext

        signal = np.load(filename, allow_pickle=False)
        signal = np.moveaxis(signal, 0, 1)
        signal = torch.from_numpy(signal.astype('float32'))
        return signal, self.utt2label[utt]


class WavDataset(Dataset):
    def __init__(self, data: list, wav_dir=WAV_DIR, cache_dir=WAV_CACHE_DIR, snr_range=(15, 30)):
        """
        :param data: List of (tone, utt, phone, start, dur)
        """
        self.data = data
        self.snr_range = snr_range
        self.wav_dir = wav_dir
        self.cache_dir = cache_dir

    def get_wav_path(self, utt: str):
        spk = utt[1:6]
        return os.path.join(self.wav_dir, spk, f'{utt}.wav')

    def get_cache_path(self, utt: str):
        spk = utt[1:6]
        spk_dir = os.path.join(self.cache_dir, spk)
        os.makedirs(spk_dir, exist_ok=True)
        return os.path.join(spk_dir, f'{utt}.npy')

    def spectro(self, y: np.ndarray, start: float, dur: float, sr=16000, fmin=50, fmax=350, hop_length=16):
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=64, n_fft=2048, hop_length=hop_length, fmin=fmin, fmax=fmax
        )
        S = librosa.power_to_db(S, ref=np.max)
        # S = S[::-1, :]  # only for visualization

        # crop to the start and the end of a phone
        end = start + dur
        s, e = librosa.time_to_frames([start, end], sr=sr, hop_length=hop_length)
        s = np.max(s, 0)
        e = np.max(e, 0)

        S = S[:, s:e + 1]
        return S

    def aug(self, y: np.ndarray, start: float, dur: float):
        from train.dataset.aug import norm_speech, add_random_noise, speed_perturb, add_random_rir

        aug_type = random.choice(['noise', 'reverb', 'sp', ''])

        if aug_type == 'noise':
            snr = random.uniform(self.snr_range[0], self.snr_range[1])
            noise_type = random.choice(['noise', 'music'])
            y = add_random_noise(y, snr, env_wav_type=noise_type)
        elif aug_type == 'sp':
            speed = random.choice([0.9, 1.1])
            y = speed_perturb(y, speed)
            start /= speed
            dur /= speed
        elif aug_type == 'reverb':
            y = add_random_rir(y)

        return norm_speech(y), start, dur

    def __getitem__(self, idx):
        tone, utt, phone, start, dur = self.data[idx]

        path = self.get_wav_path(utt)
        cache_path = self.get_cache_path(utt)
        if os.path.exists(cache_path):
            y = np.load(cache_path, allow_pickle=False)
        else:
            y, _ = librosa.load(path, sr=16000)
            np.save(cache_path, y, allow_pickle=False)

        y, start, dur = self.aug(y, start, dur)

        signal = self.spectro(y, start, dur)
        signal = np.moveaxis(signal, 0, 1)  # from (mels, time) to (time, mels)
        signal = torch.from_numpy(signal.astype('float32'))
        return signal, tone

    def __len__(self):
        return len(self.data)


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
