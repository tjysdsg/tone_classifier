import os
import random
import librosa
import numpy as np
import scipy.io.wavfile as sciwav
from scipy.signal import fftconvolve
import sox
import soundfile


def speed_perturb(y, sr=16000):
    ret = []
    for speed in [0.9, 1.1]
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


def add_noise(y, noise_signal):
    # signal, noise_signal shoud have the same len
    snr = random.uniform(0, 20)
    sigma_n = np.sqrt(10 ** (-snr / 10))
    return y + norm_speech(noise_signal) * sigma_n


def add_rir(y, rir_signal):
    rir = norm_speech(rir_signal)
    return fftconvolve(rir, y)[0 : y.shape[0]]
