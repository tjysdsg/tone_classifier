import os
import random
import librosa
import numpy as np
import scipy.io.wavfile as sciwav
from scipy.signal import fftconvolve
import sox
import soundfile

noise_list_file = '/Netdata/2017/qinxy/ASV/DeepSpeaker/egs/ffsvc/data/envir/noise_wav_list'
noise_list = []
with open(noise_list_file) as f:
    for line in f:
        noise_list.append(line.replace('\n', ''))


def speed_perturb(y, sr=16000):
    ret = []
    for speed in [0.9, 1.1]:
        tfm = sox.Transformer()
        tfm.speed(speed)
        _y = tfm.build_array(input_array=y, sample_rate_in=sr)
        ret.append(_y)
    return ret


def norm_speech(y):
    if np.std(y) == 0:
        return y
    y = (y - np.mean(y)) / np.std(y)
    return y


def add_noise(y, noise, snr):
    # make sure signal, noise_signal shoud have the same len
    l = y.shape[0]
    if noise.shape[0] == l:
        offset = 0
    else:
        offset = random.randint(0, noise.shape[0] - l)
    noise = np.array(noise[offset:offset + l])

    sigma_n = np.sqrt(10 ** (-snr / 10))
    return y + norm_speech(noise) * sigma_n


def add_random_noise(y, snr):
    noise_len = 0
    noise = None
    while noise_len < y.size:
        noise, _ = librosa.load(random.choice(noise_list), sr=16000)
        noise_len = noise.size
    return add_noise(y, noise, snr)


def add_rir(y, rir_signal):
    rir = norm_speech(rir_signal)
    return fftconvolve(rir, y)[0: y.shape[0]]
