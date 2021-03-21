import random
import librosa
import numpy as np
from scipy.signal import fftconvolve
import sox

noise_list_file = '/Netdata/2017/qinxy/ASV/DeepSpeaker/egs/ffsvc/data/envir/noise_wav_list'
noise_list = []
with open(noise_list_file) as f:
    for line in f:
        noise_list.append(line.replace('\n', ''))

rir_list_file = '/Netdata/2017/qinxy/ASV/DeepSpeaker/egs/ffsvc/data/envir/simu_rir_list'
rir_list = []
with open(rir_list_file) as f:
    for line in f:
        rir_list.append(line.replace('\n', ''))


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


def truncate_speech(signal, tlen, offset=None):
    if tlen is None:
        return signal
    if signal.shape[0] <= tlen:
        signal = np.concatenate([signal] * (tlen // signal.shape[0] + 1), axis=0)
    if offset is None:
        offset = random.randint(0, signal.shape[0] - tlen)
    return np.array(signal[offset: offset + tlen])


def add_noise(y, noise, snr):
    sigma_n = np.sqrt(10 ** (-snr / 10))
    return norm_speech(norm_speech(y) + norm_speech(noise) * sigma_n)


def add_random_noise(y, snr):
    noise, _ = librosa.load(random.choice(noise_list), sr=16000)
    noise = truncate_speech(noise, y.shape[0])
    return add_noise(y, noise, snr)


def add_rir(y, rir_signal):
    y = norm_speech(y)
    rir = norm_speech(rir_signal)
    return norm_speech(fftconvolve(rir, y)[0: y.shape[0]])


def add_random_rir(y: np.ndarray) -> np.ndarray:
    noise, _ = librosa.load(random.choice(rir_list), sr=16000)
    noise = truncate_speech(noise, y.shape[0])
    return add_rir(y, noise)


if __name__ == '__main__':
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
