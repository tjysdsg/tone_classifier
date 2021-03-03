import librosa
import random
import librosa.display
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.io.wavfile
from os.path import join as pjoin
import json
import sys
from multiprocessing import Process

# TODO: Speed perturb

data_root = '/NASdata/AudioData/mandarin/AISHELL-2/iOS/data/wav/'
# 声母
INITIALS = ['b', 'c', 'ch', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 'sh', 't', 'w', 'x', 'y', 'z', 'zh']


def chop(y, start: float, dur: float, max_dur_len=0.8):
    sr = 16000
    extra_sec = (max_dur_len - dur) / 2
    s, dur_len, extra_len, L = librosa.time_to_samples([start, dur, extra_sec, max_dur_len], sr=sr)
    ret = np.zeros(L, dtype='float32')

    fade_len = min(extra_len, librosa.time_to_samples(0.025, sr=sr))
    s = max(0, s - fade_len)
    e = s + dur_len + fade_len
    y = y[s:e]

    mask = np.zeros_like(y)
    mask[:fade_len] = np.linspace(0, 1, num=fade_len)
    mask[-fade_len:] = np.linspace(1, 0, num=fade_len)
    mask[fade_len:-fade_len] = 1
    y *= mask

    offset = (L - dur_len) // 2 - fade_len
    ret[offset:offset + y.size] = y
    return ret


def melspectrogram_feature(y, save_path: str, start: float, dur: float, fmin=50, fmax=350, extra_sec=0.1):
    sr = 16000

    plt.figure(figsize=(2.25, 2.25))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=2048, hop_length=16, fmin=fmin, fmax=fmax)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=sr, fmin=fmin, fmax=fmax, cmap='gray')

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0) 
    plt.margins(0, 0)
    plt.savefig(save_path)
    plt.close('all')


def extract_feature(data):
    from aug import add_random_noise

    tone, filename, phone, start, dur = data
    spk = filename[1:6]

    outdir = os.path.join('feats', f'{tone}')
    os.makedirs(outdir, exist_ok=True)

    # start is used to distinguish between multiple occurrence of a phone in a sentence
    outpath = pjoin(outdir, f"{filename}_{phone}_{start}.jpg")
    outpath1 = pjoin(outdir, f"{filename}_{phone}_{start}_noise.jpg")

    y = None
    try:
        if not os.path.exists(outpath) or not os.path.exists(outpath1):
            y, _ = librosa.load(pjoin(data_root, spk, f'{filename}.wav'), sr=16000)
            y = chop(y, start, dur)
        else:
            sys.stdout.write("\033[K")
            print(f"Skipping {outpath}", end='\r')

        # original
        if not os.path.exists(outpath):
            melspectrogram_feature(y, outpath, start, dur)
        else:
            sys.stdout.write("\033[K")
            print(f"Skipping {outpath}", end='\r')
    
        # data augmentation
        if not os.path.exists(outpath1):
            snr = random.uniform(50, 60)
            y_noise = add_random_noise(y, snr)
            melspectrogram_feature(y, outpath1, start, dur)
        else:
            sys.stdout.write("\033[K")
            print(f"Skipping {outpath1}", end='\r')
    except Exception as e:
        sys.stdout.write("\033[K")
        print(f"WARNING: {filename} failed\n{e}")


def collect_stats(max_dur_len=0.8):  # see durations.png
    # SSB utterance id to aishell2 utterance id
    ssb2utt = {}
    with open('ssbutt.txt') as f:
        for line in f:
            utt, ssb = line.replace('\n', '').split('|')
            ssb2utt[ssb] = utt

    # utt of filtered wavs
    filtered_wavs = set()
    with open('wav_filtered.scp') as f:
        for line in f:
            utt = line.split('\t')[0]
            filtered_wavs.add(utt)

    # utt to its phone-level transcript
    utt2trans = {}
    with open('aishell-asr-ssb-annotations.txt') as f:
        for line in f:
            tokens = line.replace('\n', '').split()
            utt = tokens[0].split('.')[0]
            utt = ssb2utt.get(utt)
            if utt is None or utt not in filtered_wavs:
                continue
            phones = tokens[2::2]
            phones = [p.replace('5', '0') for p in phones]  # 轻声 5 -> 0
            # separate initals and finals
            utt2trans[utt] = []
            for p in phones:
                if p[:2] in INITIALS:
                    utt2trans[utt].append(p[:2])
                    final = p[2:]
                elif p[:1] in INITIALS:
                    utt2trans[utt].append(p[:1])
                    final = p[1:]
                else:
                    final = p

                # 去掉儿化
                if 'er' not in final and final[-2] == 'r':
                    utt2trans[utt].append(final.replace('r', ''))   
                    utt2trans[utt].append('er0')   
                else:
                    utt2trans[utt].append(final)

    # utt to timestamps
    utt2time = {}
    with open('phone_ctm.txt') as f:
        for line in f:
            tokens = line.split()
            utt = tokens[0]
            if utt not in filtered_wavs:
                continue
            if utt not in utt2time:
                utt2time[utt] = []

            start = float(tokens[2])
            dur = float(tokens[3])
            phone = tokens[4]
            tone = phone.split('_')[-1]

            if tone.isnumeric():
                tone = int(tone)
            if phone in ['$0', 'sil', 'spn']:  # ignore empty phones
                continue

            utt2time[utt].append([start, dur, tone])

    # tone to utt, phone, start, dur
    align = {i: [] for i in range(4)}
    for k, v in utt2time.items():
        trans = utt2trans.get(k)
        if trans is None:
            continue
        if len(trans) != len(v):
            # print(f'WARNING: utt {k} different length of transcript and timestamps:\n{trans}\n{v}')
            continue
        for i, p in enumerate(trans):
            tone = p[-1]
            if not tone.isnumeric():
                continue
            tone = int(tone)
            if tone == 0:  # not including light tone
                continue
            start = v[i][0]
            dur = v[i][1]

            if dur > max_dur_len:
                continue

            if tone == v[i][2]:  # add only if ASR results are the same as annotations
                align[tone - 1].append([k, p, start, dur])  # 1st -> 4th tones starting from 0
            else:
                print(f'WARNING: utt {k} contains at least one unmatched tone')

    with open('align.json', 'w') as f:
        json.dump(align, f)


def main():
    # collect_stats()

    n_jobs = 16
    print("Extracting mel-spectrogram features")
    align = json.load(open('align.json'))
    data = [[int(label)] + vals for label, vals in align.items() for v in vals]  # list of (tone, utt, phone, start, dur)
    N = len(data)
    n_batches = N // n_jobs + 1

    for b in range(n_batches):
        print(f'Batch {b}/{n_batches}')
        offset = b * n_jobs
        ps = [Process(target=extract_feature, args=(data[offset + i],)) for i in range(n_jobs) if offset + i < N]
        for p in ps:
            p.start()
        for p in ps:
            p.join()


if __name__ == "__main__":
    main()