import random
import os
from typing import Tuple, List
import librosa
import numpy as np
from scipy.signal import fftconvolve
import sox

ENV_WAV_LIST_DIR = '/Netdata/2017/qinxy/ASV/DeepSpeaker/egs/ffsvc/data/envir/'


def get_env_wav_list(name: str) -> List[str]:
    if name == 'noise':
        filename = 'noise_wav_list'
    elif name == 'music':
        filename = 'music_wav_list'
    elif name == 'rir':
        filename = 'simu_rir_list'
    else:
        raise RuntimeError(f"Unknown environment wav list {name}")

    path = os.path.join(ENV_WAV_LIST_DIR, filename)
    ret = []
    with open(path) as f:
        for line in f:
            ret.append(line.replace('\n', ''))
    return ret


def speed_perturb(y: np.ndarray, speed: float, sr=16000) -> np.ndarray:
    tfm = sox.Transformer()
    tfm.speed(speed)
    y = tfm.build_array(input_array=y, sample_rate_in=sr)
    return y


def norm_speech(y: np.ndarray) -> np.ndarray:
    if np.std(y) == 0:
        return y
    y = (y - np.mean(y)) / np.std(y)
    return y


def truncate_speech(signal: np.ndarray, tlen: int, offset=None) -> np.ndarray:
    if tlen is None:
        return signal
    if signal.shape[0] <= tlen:
        signal = np.concatenate([signal] * (tlen // signal.shape[0] + 1), axis=0)
    if offset is None:
        offset = random.randint(0, signal.shape[0] - tlen)
    return np.array(signal[offset: offset + tlen])


def add_noise(y: np.ndarray, noise: np.ndarray, snr: float) -> np.ndarray:
    sigma_n = np.sqrt(10 ** (-snr / 10))
    return norm_speech(norm_speech(y) + norm_speech(noise) * sigma_n)


def add_random_noise(y: np.ndarray, snr: float, env_wav_type='noise') -> np.ndarray:
    noise, _ = librosa.load(random.choice(get_env_wav_list(env_wav_type)), sr=16000)
    noise = truncate_speech(noise, y.shape[0])
    return add_noise(y, noise, snr)


def add_rir(y: np.ndarray, rir_signal: np.ndarray) -> np.ndarray:
    y = norm_speech(y)
    rir = norm_speech(rir_signal)
    return norm_speech(fftconvolve(rir, y)[0: y.shape[0]])


def add_random_rir(y: np.ndarray) -> np.ndarray:
    noise, _ = librosa.load(random.choice(get_env_wav_list('rir')), sr=16000)
    noise = truncate_speech(noise, y.shape[0])
    return add_rir(y, noise)


def test():
    from sys import argv

    import librosa
    import soundfile as sf

    file = argv[1]
    y, _ = librosa.load(file, sr=16000)

    snr = float(argv[2])
    y = add_random_noise(y, snr)
    sf.write(f'noise_{snr}.wav', y, samplerate=16000, subtype='PCM_16')

    # y = add_random_rir(y)
    # sf.write(f'rir.wav', y, samplerate=16000, subtype='PCM_16')


if __name__ == '__main__':
    test()
