import numpy as np
import librosa
import os
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from train.config import WAV_DIR, CACHE_DIR, PHONE_TO_ONEHOT, SPEAKER_EMBEDDING_DIR, NUM_CLASSES
from typing import List


def collate_spectrogram(batch):
    transposed: List = list(zip(*batch))
    xs = transposed[0]
    xs: List = list(zip(*xs))
    xs = [pad_sequence(x, batch_first=True) for x in xs]  # (batch, seq_len, mels)

    y = transposed[1]
    y = torch.from_numpy(np.asarray(y, dtype='int64'))

    ret = [xs, y]
    if len(transposed) >= 3:
        durs = torch.as_tensor(transposed[2], dtype=torch.float32)  # (batch, context_size)
        ret.append(durs)

    if len(transposed) >= 4:
        onehots = torch.stack(transposed[3])  # (batch, n_phones * context_size)
        ret.append(onehots)

    if len(transposed) >= 5:
        spk_embd = torch.stack(transposed[4])  # (batch, spk_embd_size)
        ret.append(spk_embd)
    return ret


def collate_cepstrum(batch):
    transposed: List = list(zip(*batch))

    xs = transposed[0]
    xs = pad_sequence(xs, batch_first=True)  # (batch, seq_len, nfft)

    ys = transposed[1]
    lengths = [y.shape[0] for y in ys]
    ys = pad_sequence(ys, batch_first=True)  # (batch, seq_len, vocab_size)

    return [xs, ys, torch.as_tensor(lengths, dtype=torch.int)]


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
    def __init__(
            self, cache_dir=os.path.join(CACHE_DIR, 'spectro'), sr=16000, fmin=50, fmax=350, hop_length=16,
            n_fft=2048, cache=None,
    ):
        """
        :param cache: Pass in custom dictionary to shared memory across processes

        NOTE: builtin cache is not shared across threads nor processes
        """
        self.cache = cache
        if self.cache is None:
            self.cache = {}

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

        with open(self.cache_list_path) as f:
            for line in f:
                utt, path = line.replace('\n', '').split()
                self.cache[utt] = path

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

    def load_uncached(self, utt: str) -> np.ndarray:
        cache_path = self.build_cache_path(utt)
        path = get_wav_path(utt)

        y, _ = librosa.load(path, sr=16000)
        y = self.spectro(y)
        np.save(cache_path, y, allow_pickle=False)

        self.cache[utt] = cache_path
        self.cache_list_file.write(f'{utt}\t{cache_path}\n')
        return y

    def load_utt(self, utt: str) -> np.ndarray:
        cache_path = self.cache.get(utt)
        if cache_path is not None:
            try:
                y = np.load(cache_path, allow_pickle=False)
            except Exception:
                print(f'Failed to load cache at {cache_path}')
                y = self.load_uncached(utt)
        else:
            y = self.load_uncached(utt)

        return y

    def load(self, utt: str, start: float, dur: float) -> np.ndarray:
        y = self.load_utt(utt)
        y = self.chop_spectro(y, start, dur)
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


class SpeakerEmbeddingDataset(Dataset):
    def __init__(self, utts: list, data_dir=SPEAKER_EMBEDDING_DIR):
        self.utts = utts
        self.data_dir = data_dir

    def __getitem__(self, idx):
        utt = self.utts[idx]

        spk_embd = np.load(os.path.join(self.data_dir, f'{utt}.npy'))
        spk_embd = torch.from_numpy(spk_embd).type(torch.float32)
        spk_embd = spk_embd.squeeze()
        return spk_embd

    def __len__(self):
        return len(self.utts)


class SpectrogramDataset(Dataset):
    def __init__(self, data: list):
        self.extractor = CachedSpectrogramExtractor(os.path.join(CACHE_DIR, 'spectro'))
        self.data = data

    def __getitem__(self, idx):
        context = self.data[idx]
        xs = []
        for tone, utt, start, dur in context:
            if dur == 0:
                # FIXME: don't hardcode number of mels
                x = torch.zeros(1, 64)
            else:
                x = self.extractor.load(utt, start, dur)
                x = torch.from_numpy(x.astype('float32'))
            xs.append(x)

        idx = len(context) // 2  # the middle one
        return xs, context[idx][0]

    def __len__(self):
        return len(self.data)


class PhoneSegmentDataset(Dataset):
    def __init__(
            self, utt2tones: dict, include_segment_feats=False, context_size=0, include_spk=False,
            tone_pattern=None,
    ):
        self.utt2tones = utt2tones
        self.utts = list(utt2tones.keys())
        self.include_segment_feats = include_segment_feats
        self.context_size = context_size
        self.include_spk = include_spk
        self.tone_pattern = tone_pattern

        self._init_data()
        self._subset_tone_pattern()

    def _init_data(self):
        tone2idx = {t: [] for t in range(NUM_CLASSES)}
        self.utt_tone_seq = {u: [] for u in self.utts}  # used to match tone pattern, {utt -> tone_seq}

        self.flat_utts = []
        """utts of each phone sample"""
        self.data = []
        """[[tone, utt, phone, start, dur]]"""
        self.durs = []
        """[(prev_dur, next_dur), ...]"""
        self.phones = []
        """[(prev_phone, next_phone), ...]"""

        idx = 0
        for utt in self.utts:
            data = self.utt2tones[utt]

            if len(data) < 1 + 2 * self.context_size:
                continue

            prev_samples = [[-1, utt, 0, 0] for _ in range(self.context_size)]
            prev_durs = [0 for _ in range(self.context_size)]
            prev_phones = ['sil' for _ in range(self.context_size)]
            for i, (tone, phone, start, dur) in enumerate(data):
                self.utt_tone_seq[utt].append(tone)

                sample = [tone, utt, start, dur]
                tone2idx[tone].append(idx)
                self.flat_utts.append(utt)

                # set the succeeding segment of previous sample(s)
                for j in range(self.context_size):
                    if j < i:
                        self.data[-1 - j].append(sample)
                        self.durs[-1 - j].append(dur)
                        self.phones[-1 - j].append(phone)

                assert len(prev_samples) == len(prev_durs) == len(prev_phones) == self.context_size
                self.data.append(prev_samples + [sample])
                self.durs.append(prev_durs + [dur])
                self.phones.append(prev_phones + [phone])

                if self.context_size:
                    prev_samples.pop(0)
                    prev_samples.append(sample)
                    prev_durs.pop(0)
                    prev_durs.append(dur)
                    prev_phones.pop(0)
                    prev_phones.append(phone)

                idx += 1

            # set the succeeding segment of the last few samples
            for j in range(self.context_size):
                self.data[-1 - j] += [[-1, utt, 0, 0] for _ in range(self.context_size - j)]
                self.durs[-1 - j] += [0 for _ in range(self.context_size - j)]
                self.phones[-1 - j] += ['sil' for _ in range(self.context_size - j)]

        assert len(self.data) == len(self.durs) == len(self.phones) == len(self.flat_utts)

        self._dataset = SpectrogramDataset(self.data)
        if self.include_spk:
            self.spk_dataset = SpeakerEmbeddingDataset(self.flat_utts)

        """Balancing data (mostly removing initials)"""
        size = len(self.data)
        for t, indices in tone2idx.items():
            if t != 5:  # neutral tone simply contains too little data
                size = min(size, len(indices))

        print(f'Balanced data size: {NUM_CLASSES} * {size}')

        self.indices = []
        for t in tone2idx.keys():
            np.random.shuffle(tone2idx[t])
            self.indices += tone2idx[t][:size]
        np.random.shuffle(self.indices)

    def _subset_tone_pattern(self):
        if self.tone_pattern is None:
            return

        indices = []
        idx = 0
        for utt, seq in self.utt_tone_seq.items():  # type: str, List[int]
            # FIXME: brute force matching
            n = len(self.tone_pattern)
            seq_len = len(seq)
            for i in range(seq_len - n):
                tmp = []
                matched = True
                t = 0
                j = i
                while t < n and j < seq_len:
                    if seq[j] == 0:  # skip no-tone
                        j += 1
                        continue

                    if seq[j] != self.tone_pattern[t]:
                        matched = False
                        break

                    tmp.append(idx + j)

                    t += 1
                    j += 1

                if matched and len(tmp) == n:
                    indices += tmp
            idx += seq_len

        self.indices = indices
        np.random.shuffle(self.indices)
        print(f'Number of samples that matches the tone pattern: {len(self.indices)}')

    def __getitem__(self, idx):
        idx = self.indices[idx]

        spectro, y = self._dataset[idx]
        ret = [spectro, y]

        if self.include_segment_feats:
            # durations
            ret.append(self.durs[idx])

            # onehot encodings
            onehot = [
                torch.from_numpy(PHONE_TO_ONEHOT[ph]).type(torch.float32)
                for ph in self.phones[idx]
            ]
            ret.append(torch.cat(onehot))

        if self.include_spk:
            ret.append(self.spk_dataset[idx])
        return ret

    def __len__(self):
        return len(self.indices)


class CepstrumDataset(Dataset):
    def __init__(self, text: str, wavscp: str, nfft=512, sr=16000):
        from train.utils import load_utt2seq

        self.sr = sr
        self.nfft = nfft

        self.utt2tones = load_utt2seq(text, int)
        self.utt2path = load_utt2seq(wavscp)
        self.utts = list(set(self.utt2tones.keys()) & set(self.utt2path.keys()))

    def __getitem__(self, idx):
        from train.dataset.cepstrum import cepstrum

        utt = self.utts[idx]
        y, _ = librosa.load(self.utt2path[utt][0], sr=self.sr)
        ceps = cepstrum(y, self.sr, nfft=self.nfft)
        ceps = np.asarray(ceps, dtype='float32')

        tones = np.asarray(self.utt2tones[utt], dtype='int') + 1  # plus 1 because <blank>

        return torch.from_numpy(ceps), torch.from_numpy(tones)

    def __len__(self):
        return len(self.utts)


def create_dataloader(
        utts: list, utt2tones: dict, include_segment_feats=False,
        include_spk=False, context_size=0, batch_size=64, n_workers=10,
        tone_pattern=None,
):
    u2t = {u: utt2tones[u] for u in utts}

    # count the number of each tone
    tones = {t: 0 for t in range(NUM_CLASSES)}
    for _, t in u2t.items():
        for d in t:
            tone = d[0]
            tones[tone] += 1
    print(tones)

    return DataLoader(
        PhoneSegmentDataset(
            u2t, include_segment_feats=include_segment_feats, context_size=context_size, include_spk=include_spk,
            tone_pattern=tone_pattern,
        ),
        batch_size=batch_size, num_workers=n_workers, collate_fn=collate_spectrogram,
    )
