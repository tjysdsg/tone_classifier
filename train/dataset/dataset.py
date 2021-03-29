import numpy as np
import librosa
import os
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from train.config import WAV_DIR, CACHE_DIR, PHONE_TO_ONEHOT
from typing import List
import random


def collate_spectrogram(batch):
    transposed: List = list(zip(*batch))
    x = transposed[0]
    y = transposed[1]

    y = torch.from_numpy(np.asarray(y, dtype='int64'))
    x = pad_sequence(x, batch_first=True)  # (batch, seq_len, mels)

    ret = [x, y]
    if len(transposed) >= 3:
        durs = torch.as_tensor(transposed[2], dtype=torch.float32)  # (batch, 3)
        ret.append(durs)

    if len(transposed) >= 4:
        # TODO: previous onehot, curr onehot, next onehot
        onehots = torch.stack(transposed[3])  # (batch, n_phones)
        ret.append(onehots)

    return ret


def pad_seq(labels: List[torch.Tensor], padding_value=0) -> torch.Tensor:
    max_len = np.max([e.shape[0] for e in labels])
    ret = torch.full((len(labels), max_len), padding_value)
    for i, e in enumerate(labels):
        len_e = e.shape[0]
        ret[i, :len_e] = e
    return ret


def collate_sequential_spectrogram(batch):
    transposed: list = list(zip(*batch))
    spectrograms: List[SpectroFeat] = transposed[0]  # (batch_size, seq_len, ...)
    x = [[e, e.spectro.shape[0]] for e in spectrograms]

    y = transposed[1]  # (batch_size, seq_len)
    y = pad_seq(y, padding_value=-100)  # -100 is ignored by NLLLoss

    return x, torch.as_tensor(y, dtype=torch.long)


def get_spk_from_utt(utt: str):
    return utt[:7]


def get_wav_path(utt: str):
    spk = get_spk_from_utt(utt)
    return os.path.join(WAV_DIR, spk, f'{utt}.wav')


class SpectroFeat:
    def __init__(self, spectro: torch.Tensor, lengths: List[int or float] = None, onehots: torch.Tensor = None):
        """
        :param spectro: Spectrograms
        :param lengths: List of signal lengths
        """
        self.spectro = spectro
        self.lengths = lengths
        self.onehots = onehots


class CachedSpectrogramExtractor:
    def __init__(self, cache_dir: str, sr=16000, fmin=50, fmax=350, hop_length=16, n_fft=2048):
        self.sr = sr
        self.fmin = fmin
        self.fmax = fmax
        self.hop_length = hop_length
        self.n_fft = n_fft

        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_list_path = os.path.join(self.cache_dir, 'wav.scp')

        f = open(self.cache_list_path, 'a')  # `touch wav.scp`
        f.close()

        self.cache = {}
        with open(self.cache_list_path) as f:
            for line in f:
                spectro_id, path = line.replace('\n', '').split()
                self.cache[spectro_id] = path

        self.cache_list_file = open(self.cache_list_path, 'a', buffering=1)  # line buffered

    def build_cache_path(self, utt: str):
        spk = get_spk_from_utt(utt)
        spk_dir = os.path.join(self.cache_dir, spk)
        os.makedirs(spk_dir, exist_ok=True)
        return os.path.join(spk_dir, f'{utt}.npy')

    def spectro(self, y: np.ndarray):
        S = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_mels=64, n_fft=self.n_fft, hop_length=self.hop_length, fmin=self.fmin, fmax=self.fmax
        )
        S = librosa.power_to_db(S, ref=np.max)
        return np.asarray(S, dtype='float32')

    def chop_spectro(self, S: np.ndarray, start: float, dur: float):
        # crop to the start and the end of a phone
        end = start + dur
        s, e = librosa.time_to_frames([start, end], sr=self.sr, hop_length=self.hop_length)
        s = np.max(s, 0)
        e = np.max(e, 0)
        S = S[:, s:e + 1]
        S = np.moveaxis(S, 0, 1)  # from (mels, time) to (time, mels)
        return S

    def load_uncached(self, utt: str, start: float, dur: float) -> np.ndarray:
        cache_path = self.build_cache_path(utt)
        path = get_wav_path(utt)

        y, _ = librosa.load(path, sr=16000)
        y = self.spectro(y)
        np.save(cache_path, y, allow_pickle=False)

        y = self.chop_spectro(y, start, dur)

        self.cache[utt] = cache_path
        self.cache_list_file.write(f'{utt}\t{cache_path}\n')
        return y

    def load(self, utt: str, start: float, dur: float) -> np.ndarray:
        cache_path = self.cache.get(utt)
        if cache_path is not None:
            try:
                y = np.load(cache_path, allow_pickle=False)
                y = self.chop_spectro(y, start, dur)
            except ValueError:
                print(f'Failed to load cache at {cache_path}')
                y = self.load_uncached(utt, start, dur)
        else:
            y = self.load_uncached(utt, start, dur)

        return y

    """
    def aug(self, y: np.ndarray, start: float, dur: float):
        import random
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
    """


class SpectrogramDataset(Dataset):
    def __init__(self, utt2tones: dict, include_dur=False, include_onehot=False):
        self.utt2tones = utt2tones
        self.utts = list(utt2tones.keys())
        self.extractor = CachedSpectrogramExtractor(os.path.join(CACHE_DIR, 'spectro'))
        self.include_dur = include_dur
        self.include_onehot = include_onehot

        self.data = []
        self.durs = []
        """[(prev_dur, next_dur), ...]"""
        self.phones = []
        """[(prev_phone, next_phone), ...]"""
        for utt in self.utts:
            data = self.utt2tones[utt]
            prev_dur = 0  # prev_dur of the first segment in a sentence is 0
            prev_phone = 'sil'  # prev_phone of the first segment in a sentence is 'sil'
            for i, (tone, phone, start, dur) in enumerate(data):
                if i > 0:
                    # set next_dur of the previous segment to dur
                    self.durs[-1].append(dur)
                    # set next_phone of the previous segment to phone
                    self.phones[-1].append(phone)

                self.data.append([tone, utt, phone, start, dur])

                self.durs.append([
                    prev_dur,  # next_dur,
                ])
                self.phones.append([
                    prev_phone,  # next_phone,
                ])
                prev_dur = dur
                prev_phone = phone
            self.durs[-1].append(0)  # set next_dur of the last segment to 0
            self.phones[-1].append('sil')  # set next_phone of the last segment to 'sil'

        assert len(self.data) == len(self.durs) == len(self.phones)

        self.size = len(self.data)
        self.argshuffle = np.random.permutation(self.size)

    def __getitem__(self, idx):
        idx = self.argshuffle[idx]
        tone, utt, phone, start, dur = self.data[idx]

        x = self.extractor.load(utt, start, dur)
        x = torch.from_numpy(x.astype('float32'))
        ret = [x, tone]

        if self.include_dur:
            prev_dur, next_dur = self.durs[idx]
            ret.append([prev_dur, dur, next_dur])

        if self.include_onehot:
            prev_phone, next_phone = self.phones[idx]
            onehot = [
                torch.from_numpy(PHONE_TO_ONEHOT[prev_phone]).type(torch.float32),
                torch.from_numpy(PHONE_TO_ONEHOT[phone]).type(torch.float32),
                torch.from_numpy(PHONE_TO_ONEHOT[next_phone]).type(torch.float32),
            ]
            ret.append(torch.cat(onehot))

        return ret

    def __len__(self):
        return self.size


class PhoneSegmentDataset(Dataset):
    def __init__(self, *args, feat_type='spectrogram', **kwargs):
        self.feat_type = feat_type
        # if self.feat_type == 'spectrogram':
        self._dataset = SpectrogramDataset(*args, **kwargs)

    def __getitem__(self, idx):
        return self._dataset[idx]

    def __len__(self):
        return len(self._dataset)


class SequentialSpectrogramDataset(Dataset):
    def __init__(self, utt2tones: dict, include_dur=False):
        self.utts = list(utt2tones.keys())
        self.utt2tones = utt2tones
        self.extractor = CachedSpectrogramExtractor(os.path.join(CACHE_DIR, 'spectro'))
        self.include_dur = include_dur

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
        lengths = []
        for tone, phone, start, dur in seq:
            x = self.extractor.load(utt, start, dur)

            if self.include_dur:
                lengths.append(x.shape[0])

            x = torch.from_numpy(x.astype('float32'))
            xs.append(x)
            ys.append(tone)

        x = pad_sequence(xs, batch_first=True)  # (seq_len, sig_len, mels)
        y = torch.as_tensor(ys, dtype=torch.long)  # (seq_len,)

        feat = SpectroFeat(x)
        if self.include_dur:
            feat.lengths = lengths

        return feat, y
