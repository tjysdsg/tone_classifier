import librosa
import librosa.display
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.io.wavfile
from pydub import AudioSegment
from os.path import join as pjoin
import json


data_root = '/NASdata/AudioData/mandarin/AISHELL-2/iOS/data/wav/'
# 声母
INITIALS = ['b', 'c', 'ch', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 'sh', 't', 'w', 'x', 'y', 'z', 'zh']


def melspectrogram_feature(audio_path: str, save_path: str, start: float, end: float, fmin=50, fmax=350, extra_sec=0.1):
    y, sr = librosa.load(audio_path, sr=16000)
    start, end = librosa.time_to_samples([start - extra_sec, start + end + extra_sec], sr=sr)
    if start < 0:
        start = 0
    y = y[start:end]
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=2048, hop_length=16, fmin=fmin, fmax=fmax)

    plt.figure(figsize=(2.25, 2.25))
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=sr, fmin=fmin, fmax=fmax, cmap='magma')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0) 
    plt.margins(0, 0)
    plt.savefig(save_path)
    plt.close('all')


def extract_feature_for_tone(tone: int, configs):
    outdir = os.path.join('feats', f'{tone}')
    os.makedirs(outdir, exist_ok=True)

    n = len(configs)
    prev_prog = 0
    for j, e in enumerate(configs):
        prog = int(100 * j / n)
        if prog != prev_prog:
            print(f'tone {tone}: {j}/{n}')
            prev_prog = prog
        filename, phone, start, dur = e
        spk = filename[1:6]

        outpath = pjoin(outdir, f"{j}_{filename}_{phone}.jpg")
        if os.path.exists(outpath):
            continue
        melspectrogram_feature(pjoin(data_root, spk, f'{filename}.wav'), outpath, start, dur)


if __name__ == "__main__":
    """
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
    # json.dump(utt2trans, open('utt2trans.json', 'w'))

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
            if phone in ['$0', 'sil']:  # ignore empty phones
                continue
            utt2time[utt].append([start, dur])
    for k, v in utt2time.items():  # sort by start time
        v.sort(key=lambda x: x[0])

    # tone to utt, phone, start, dur
    align = {i: [] for i in range(5)}
    for k, v in utt2time.items():
        trans = utt2trans.get(k)
        if trans is None:
            continue
        if len(trans) != len(v):
            print(f'WARNING: utt {k} different length of transcript and timestamps:\n{trans}\n{v}')
            continue
        for i, p in enumerate(trans):
            tone = p[-1]
            if not tone.isnumeric():
                continue
            tone = int(tone)
            align[tone].append([k, p, v[i][0], v[i][1]])

    json.dump(align, open('align.json', 'w'))
    """

# """
    print("Extracting mel-spectrogram features")
    align = json.load(open('align.json'))
    align = {int(k): v for k,v in align.items()}

    from multiprocessing import Process
    ps = [Process(target=extract_feature_for_tone, args=(i, align[i])) for i in range(5)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
# """