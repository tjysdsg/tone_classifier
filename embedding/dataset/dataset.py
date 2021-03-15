import librosa
import numpy as np
import os
import random
import scipy.io.wavfile as sciwav
import torch
import torchaudio
from python_speech_features import sigproc
from scipy.signal import fftconvolve
from torch.utils.data import Dataset


class WavDataset(Dataset):
    def __init__(self, wav_scp, utt2label=None,
                 fs=16000, preemph=0.97, channel=None,
                 is_aug=False, snr=None, noise_list=None):
        self.wav_scp = wav_scp
        self.utt2label = utt2label
        self.channel = channel

        self.fs = fs
        self.preemph = preemph

        self.is_aug = is_aug
        self.noise_list = noise_list
        self.snr = snr

    def __len__(self):
        return len(self.wav_scp)

    def _load_data(self, filename):
        if os.path.splitext(filename)[-1] == '.wav':
            fs, signal = sciwav.read(filename, mmap=True)
        elif os.path.splitext(filename)[-1] == '.m4a':
            signal, fs = librosa.load(filename, sr=self.fs)
        if fs != self.fs:
            signal, fs = librosa.load(filename, sr=self.fs)
        if len(signal.shape) == 2 and self.channel:
            channel = random.choice(self.channel) if type(self.channel) == list else self.channel
            return signal[:, channel]
        return signal

    def _norm_speech(self, signal):
        if np.std(signal) == 0:
            return signal
        # signal = signal / (np.abs(signal).max())
        signal = (signal - np.mean(signal)) / np.std(signal)
        return signal

    def _augmentation(self, signal, filename):
        signal = self._norm_speech(signal)

        noise_types = random.choice(['reverb', 'sox', 'noise'])

        if noise_types == 'spec_aug':
            return signal, 1  # indicator to apply specAug at feature calculator

        elif noise_types == 'sox':
            E = torchaudio.sox_effects.SoxEffectsChain()
            effect = random.choice(['tempo', 'vol'])
            if effect == 'tempo':
                E.append_effect_to_chain("tempo", random.choice([0.9, 1.1]))
            elif effect == 'vol':
                E.append_effect_to_chain("vol", random.random() * 15 + 5)
            E.append_effect_to_chain("rate", self.fs)
            E.set_input_file(filename)
            signal_sox, _ = E.sox_build_flow_effects()
            return self._truncate_speech(signal_sox.numpy()[0], len(signal)), 0

        elif noise_types == 'reverb':
            rir = self._norm_speech(self._load_data(random.choice(self.noise_list[noise_types])))
            return fftconvolve(rir, signal)[0: signal.shape[0]], 0

        else:
            noise_signal = np.zeros(signal.shape[0], dtype='float32')
            for noise_type in random.choice([['noise'], ['music'], ['babb', 'music'], ['babb'] * random.randint(3, 8)]):
                noise = self._load_data(random.choice(self.noise_list[noise_type]))
                noise = self._truncate_speech(noise, signal.shape[0])
                noise_signal = noise_signal + self._norm_speech(noise)
            snr = random.uniform(self.snr[0], self.snr[1])
            sigma_n = np.sqrt(10 ** (- snr / 10))
            return signal + self._norm_speech(noise_signal) * sigma_n, 0

    def _truncate_speech(self, signal, tlen, offset=None):
        if tlen == None:
            return signal
        if signal.shape[0] <= tlen:
            signal = np.concatenate([signal] * (tlen // signal.shape[0] + 1), axis=0)
        if offset == None:
            offset = random.randint(0, signal.shape[0] - tlen)
        return np.array(signal[offset: offset + tlen])

    def __getitem__(self, idx):
        if isinstance(idx, int):
            tlen = None
        elif len(idx) == 2:
            idx, tlen = idx
            tlen = int(tlen * self.fs)
        else:
            raise AssertionError("The idx should be int or a list with length of 2.")

        utt, filename = self.wav_scp[idx]
        signal = self._load_data(filename)

        offset = None if self.utt2label else 0
        signal = self._truncate_speech(signal, tlen, offset)

        is_spec_aug = 0
        if self.utt2label and self.is_aug and random.choice([0, 1, 1]):
            # only do data augmentation at training (with utt2label)
            # 2/3 data augmentation; 1/3 clean data
            signal, is_spec_aug = self._augmentation(signal, filename)

        signal = self._norm_speech(signal)
        signal = sigproc.preemphasis(signal, self.preemph)
        signal = torch.from_numpy(signal.astype('float32'))

        if self.utt2label:
            return signal, is_spec_aug, self.utt2label[utt]
        else:
            return signal, utt


class EmbdDataset(Dataset):
    """
    This dataset is for speaker diarization training. 
    embd_scp: {utt: embedding_file}, the utt is the id of the segments in subsegments_data
    spk2utt: {spk: list[utts]}, where the spk is not the real speaker. It is the id of a complete conversation audio file.
    labels: {utt: label}, where the label is the speaker id.
    """

    def __init__(self, embd_scp, reco2utt, labels, pre_load=True):
        from tqdm import tqdm
        from functools import reduce

        self.embd_scp = dict(embd_scp)
        self.reco2utt = {i[0]: i[1:] for i in reco2utt}
        self.utt2label = dict(labels)
        self.pre_load = pre_load

        recos = sorted(list(self.reco2utt.keys()))
        utts = reduce(lambda a, b: a + b, [self.reco2utt[reco] for reco in recos])
        self.repeated_recos = reduce(lambda a, b: a + b, [[reco] * len(self.reco2utt[reco]) for reco in recos])
        self.len = len(self.repeated_recos)

        if self.pre_load:
            self.utt2npy = dict()
            for utt in tqdm(utts):
                if utt in self.embd_scp:
                    self.utt2npy[utt] = np.load(self.embd_scp[utt]).reshape(-1)
                else:
                    print(utt)

    def truncateUtts(self, utts, tlen):
        if tlen == None:
            return utts
        if len(utts) < tlen:
            utts = utts * math.ceil(tlen / len(utts))
        offset = random.randint(0, len(utts) - tlen)
        return utts[offset:offset + tlen]

    def __getitem__(self, idx):
        if isinstance(idx, int):
            tlen = None
        elif len(idx) == 2:
            idx, tlen = idx
        else:
            raise AssertionError("The idx should be int or list with lenght of 2.")

        reco = self.repeated_recos[idx]
        utts = self.reco2utt[reco]
        utt1 = np.random.choice(utts)
        utt2s = self.truncateUtts(utts, tlen)

        if self.pre_load:
            embd1 = self.utt2npy[utt1]
        else:
            embd1 = np.load(self.embd_scp[utt1]).reshape(-1)
        embd1 = torch.from_numpy(embd1)
        embd1s = embd1.unsqueeze(0).repeat(len(utt2s), 1)
        if self.pre_load:
            embd2s = [self.utt2npy[utt] for utt in utt2s]
        else:
            embd2s = [np.load(self.embd_scp[utt]).reshape(-1) for utt in utt2s]
        embd2s = np.stack(embd2s, axis=0)
        embd2s = torch.from_numpy(embd2s)
        feat = torch.cat([embd1s, embd2s], dim=1)

        label1 = self.utt2label[utt1]
        target = [label1 == self.utt2label[utt] for utt in utt2s]
        target = torch.tensor(target)

        return feat, target

    def __len__(self):
        return self.len
