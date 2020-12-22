import librosa
import librosa.display
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.io.wavfile
from pydub import AudioSegment
from os.path import join as pjoin
import os
import json

data_root = '/NASdata/AudioData/mandarin/AISHELL-2/iOS/data/wav/'
# 声母
INITIALS = ['b', 'c', 'ch', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 'sh', 't', 'w', 'x', 'y', 'z', 'zh']


def time_range_to_file(inpath: str, outpath: str, start: float, duration: float):
    wav = AudioSegment.from_wav(inpath)

    end = start + duration
    start *= 1000  # milliseconds
    end *= 1000  # milliseconds

    seg = wav[start:end]
    seg.export(outpath, format="wav", parameters=["-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000"])


def melspectrogram_feature(audio_path, save_path, fmin=50, fmax=350):
    sr, y = scipy.io.wavfile.read(audio_path)
    if y.dtype is not np.float:
        y = y.astype('float32') / 32767
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=2048, hop_length=16, fmin=fmin, fmax=fmax)

    plt.figure(figsize=(2.25, 2.25))
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=sr, fmin=fmin, fmax=fmax)

    plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
    plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0) 
    plt.margins(0, 0)
    name = os.path.basename(audio_path).split('.')[0] + '.jpg'
    plt.savefig(os.path.join(save_path, name))
    plt.close('all')

    return os.path.join(save_path, name)


if __name__ == "__main__":
    # SSB utterance id to aishell2 utterance id
    ssb2utt = {}
    with open('ssbutt.txt') as f:
        for line in f:
            utt, ssb = line.replace('\n', '').split('|')
            ssb2utt[ssb] = utt

    # utt to its phone-level transcript
    utt2trans = {}
    with open('aishell-asr-ssb-annotations.txt') as f:
        for line in f:
            tokens = line.replace('\n', '').split()
            utt = tokens[0].split('.')[0]
            if utt not in ssb2utt:
                continue
            utt = ssb2utt[utt]
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
        if k not in utt2trans:
            continue
        trans = utt2trans[k]
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
    align = json.load(open('align.json'))
    align = {int(k): v for k,v in align.items()}

    for i in range(5)[::-1]:
        aligndir = pjoin('data', f'{i}')
        os.makedirs(aligndir, exist_ok=True)
        outdir = os.path.join('feats', f'{i}')
        os.makedirs(outdir, exist_ok=True)

        n = len(align[i])
        for j, e in enumerate(align[i]):
            print(f'tone {i}: {j}/{n}', end='\r')
            filename, phone, start, dur = e
            spk = filename[1:6]

            wavpath = pjoin(aligndir, f"{j}_{filename}_{phone}.wav")
            time_range_to_file(pjoin(data_root, spk, f'{filename}.wav'), wavpath, start, dur)
            melspectrogram_feature(wavpath, outdir)
    """