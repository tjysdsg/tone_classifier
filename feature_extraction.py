import librosa
import librosa.display
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.io.wavfile
from os.path import join as pjoin
import json


data_root = '/NASdata/AudioData/mandarin/AISHELL-2/iOS/data/wav/'
# 声母
INITIALS = ['b', 'c', 'ch', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 'sh', 't', 'w', 'x', 'y', 'z', 'zh']


def chop(audio_path: str, start: float, dur: float):
    extra_sec = (0.8 - dur) / 2
    y, sr = librosa.load(audio_path, sr=16000)
    extra_len = librosa.time_to_samples(extra_sec, sr=sr)
    dur_len = librosa.time_to_samples(dur, sr=sr)

    start, end = librosa.time_to_samples([start - extra_sec, start + dur + extra_sec], sr=sr)
    pad_l = 0
    pad_r = 0
    if start < 0:
        pad_l = -start
        start = 0
    if end > y.size:
        pad_r = end - y.size
        end = y.size
    y = np.pad(y, (pad_l, pad_r), 'constant', constant_values=(0, 0))
    y = y[start:end]

    l = extra_len
    r = extra_len + dur_len
    fade_len = min(extra_len, librosa.time_to_samples(0.1, sr=sr))
    mask = np.zeros_like(y)
    mask[l-fade_len:l] = np.linspace(0, 1, num=fade_len)
    # FIXME: rhs has larger dimension than lhs
    mask[r:r+fade_len] = np.linspace(1, 0, num=fade_len)
    mask[l:r] = 1
    y *= mask
    return y


def melspectrogram_feature(audio_path: str, save_path: str, start: float, dur: float, fmin=50, fmax=350, extra_sec=0.1):
    y = chop(audio_path, start, dur)
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


def extract_feature_for_tone(tone: int, configs):
    outdir = os.path.join('feats', f'{tone}')
    os.makedirs(outdir, exist_ok=True)

    # n = len(configs)
    n = 50000
    prev_prog = 0
    dotlist_file = open(os.path.join('feats', f'{tone}.list'), 'w')
    for j, e in enumerate(configs):
        prog = int(100 * j / n)
        if prog != prev_prog:
            print(f'tone {tone}: {j}/{n}')
            prev_prog = prog

        if j >= n:
            break

        filename, phone, start, dur = e
        spk = filename[1:6]

        # start is used to distinguish between multiple occurrence of a phone in a sentence
        outpath = pjoin(outdir, f"{filename}_{phone}_{start}.jpg")
        if os.path.exists(outpath):
            continue
        try:
            melspectrogram_feature(pjoin(data_root, spk, f'{filename}.wav'), outpath, start, dur)
        except:
            print(f"WARNING: {filename} failed")
            return

        # write to feats/*.list
        dotlist_file.write(outpath + '\n')
    dotlist_file.close()


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
            utt = tokens[0]  # TODO: sp0.9-* and sp1.1-*
            if utt not in filtered_wavs:
                continue
            if utt not in utt2time:
                utt2time[utt] = []
            start = float(tokens[2])
            dur = float(tokens[3])
            if dur > 0.8:  # see durations.png
                continue
            phone = tokens[4]
            tone = phone.split('_')[-1]
            if tone.isnumeric():
                tone = int(tone)
            if phone in ['$0', 'sil']:  # ignore empty phones
                continue
            utt2time[utt].append([start, dur, tone])
    for k, v in utt2time.items():  # sort by start time
        v.sort(key=lambda x: x[0])

    # all_dur = []

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

            if tone == v[i][2]:  # add only if ASR results are the same as annotations
                align[tone - 1].append([k, p, start, dur])  # 1st -> 4th tones starting from 0
            else:
                print(f'WARNING: utt {k} contains at least one unmatched tone')
            all_dur.append(dur)

    json.dump(align, open('align.json', 'w'))
    # plt.plot(all_dur, 'o')
    # plt.savefig('durations.png')
    """

    # """
    print("Extracting mel-spectrogram features")
    align = json.load(open('align.json'))
    align = {int(k): v for k,v in align.items()}

    from multiprocessing import Process
    ps = [Process(target=extract_feature_for_tone, args=(i, align[i])) for i in range(4)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    # """