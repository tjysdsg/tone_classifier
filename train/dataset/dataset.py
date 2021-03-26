import numpy as np
import librosa
import random
import os
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from train.config import WAV_DIR, CACHE_DIR, PRETRAINED_EMBEDDINGS_DIR


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
    def __init__(self, data: list, wav_dir=WAV_DIR, cache_dir=os.path.join(CACHE_DIR, 'spectro'), snr_range=(20, 50)):
        """
        :param data: List of (tone, utt, phone, start, dur)
        """
        self.data = data
        self.snr_range = snr_range
        self.wav_dir = wav_dir
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_list_path = os.path.join(self.cache_dir, 'wav.scp')

        self.cache = {}
        with open(self.cache_list_path) as f:
            for line in f:
                utt, path = line.replace('\n', '').split()
                self.cache[utt] = path

        self.cache_list_file = open(self.cache_list_path, 'a', buffering=1)  # line buffered

    def get_spk_from_utt(self, utt: str):
        return utt[:7]

    def get_wav_path(self, utt: str):
        spk = self.get_spk_from_utt(utt)
        return os.path.join(self.wav_dir, spk, f'{utt}.wav')

    def build_cache_path(self, utt: str):
        spk = self.get_spk_from_utt(utt)
        spk_dir = os.path.join(self.cache_dir, spk)
        os.makedirs(spk_dir, exist_ok=True)
        return os.path.join(spk_dir, f'{utt}.npy')

    def spectro(self, y: np.ndarray, start: float, dur: float, sr=16000, fmin=50, fmax=350, hop_length=16, n_fft=2048):
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=64, n_fft=n_fft, hop_length=hop_length, fmin=fmin, fmax=fmax
        )
        S = librosa.power_to_db(S, ref=np.max)

        # crop to the start and the end of a phone
        end = start + dur
        s, e = librosa.time_to_frames([start, end], sr=sr, hop_length=hop_length)
        s = np.max(s, 0)
        e = np.max(e, 0)

        return S[:, s:e + 1]

    def aug(self, y: np.ndarray, start: float, dur: float):
        from train.dataset.aug import norm_speech, add_random_noise, speed_perturb, add_random_rir

        aug_type = random.choice(['noise', 'reverb', 'sp', ''])

        if aug_type == 'noise':
            snr = random.uniform(self.snr_range[0], self.snr_range[1])
            noise_type = random.choice(['noise', 'music'])
            y = add_random_noise(norm_speech(y), snr, env_wav_type=noise_type)
        elif aug_type == 'sp':
            speed = random.choice([0.9, 1.1])
            y = speed_perturb(y, speed)
            y = norm_speech(y)
            start /= speed
            dur /= speed
        elif aug_type == 'reverb':
            y = add_random_rir(norm_speech(y))

        return y, start, dur

    def __getitem__(self, idx):
        tone, utt, phone, start, dur = self.data[idx]

        cache_path = self.cache.get(utt)
        if cache_path is not None:
            y = np.load(cache_path, allow_pickle=False)
        else:
            cache_path = self.build_cache_path(utt)
            path = self.get_wav_path(utt)

            y, _ = librosa.load(path, sr=16000)
            y = self.spectro(y, start, dur)
            y = np.moveaxis(y, 0, 1)  # from (mels, time) to (time, mels)

            np.save(cache_path, y, allow_pickle=False)
            self.cache[utt] = cache_path
            self.cache_list_file.write(f'{utt}\t{cache_path}\n')

        y = torch.from_numpy(y.astype('float32'))
        return y, tone

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
        utt, seq = self.sequences[idx]

        xs = []
        ys = []
        for tone, phone, start, dur in seq:
            # FIXME
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
    def __init__(self, utts: list, utt2tones: dict, embedding_dir=PRETRAINED_EMBEDDINGS_DIR):
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
