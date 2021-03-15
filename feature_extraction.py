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
INITIALS = ['b', 'c', 'ch', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 'sh', 't', 'w', 'x', 'y',
            'z', 'zh']


def spectro(y, start: float, dur: float, max_dur=0.6, sr=16000, fmin=50, fmax=350, hop_length=16):
    from skimage.transform import resize

    # make a general bounding box and crop it
    extra_dur = (max_dur - dur) / 2
    bound_s = start - extra_dur
    bound_e = start + dur + extra_dur
    bs, be = librosa.time_to_samples([bound_s, bound_e], sr=sr)
    y = y[bs:be]

    # mel-spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=2048, hop_length=hop_length, fmin=fmin, fmax=fmax)
    S = librosa.power_to_db(S, ref=np.max)
    S = S[::-1, :]

    # then crop precisely to the start and the end of a phone
    # start = np.max([extra_dur - 0.05, 0])
    # end = np.min([start + dur + 0.05, max_dur])
    start = extra_dur
    end = start + dur
    s, e = librosa.time_to_frames([start, end], sr=sr, hop_length=hop_length)
    s = np.max(s, 0)
    e = np.max(e, 0)
    S[:, :s] = 0
    S[:, e + 1:] = 0

    # resize
    S = resize(
        S, output_shape=(225, 225),
        anti_aliasing=False, preserve_range=True, clip=False,
        mode='constant', cval=0, order=0,
    )
    return S


def plot_spectro_to_file(S, output_path: str, cmap='gray'):
    plt.imshow(S, cmap=cmap)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(output_path)
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
        else:
            sys.stdout.write("\033[K")
            print(f"Skipping {outpath}", end='\r')

        # original
        if not os.path.exists(outpath):
            S = spectro(y, start, dur)
            plot_spectro_to_file(S, outpath)
        else:
            sys.stdout.write("\033[K")
            print(f"Skipping {outpath}", end='\r')

        # data augmentation
        if not os.path.exists(outpath1):
            snr = random.uniform(50, 60)
            y_noise = add_random_noise(y, snr)
            S = spectro(y, start, dur)
            plot_spectro_to_file(S, outpath1)
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
        prev_utt = 'fuck'
        prev_dur = 0
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

            # triphone
            # NOTE: don't change the value of dur, it's the true duration of this phone
            # FIXME: what to do when no previous phone
            if utt != prev_utt:
                utt2time[utt].append([start, dur, tone])
            else:
                # include this phone in the previous triphone
                utt2time[utt][-1][1] += dur / 2

                # include previous phone in this triphone
                utt2time[utt].append(
                    [start - prev_dur / 2, prev_dur / 2 + dur, tone]
                )
            prev_dur = dur
            prev_utt = utt

    align = {i: [] for i in range(4)}  # tone to utt, phone, start, dur
    all_data = []  # list of (tone, utt, phone, start, dur)
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
                # `tone` starts at 0
                align[tone - 1].append([k, p, start, dur])
                all_data.append([tone - 1, k, p, start, dur])
            else:
                print(f'WARNING: utt {k} contains at least one unmatched tone')

    with open('align.json', 'w') as f:
        json.dump(align, f)

    with open('all_data.json', 'w') as f:
        json.dump(all_data, f)


def main():
    # collect_stats()

    print("Loading...")
    n_jobs = 16
    data = json.load(open('all_data.json'))  # list of (tone, utt, phone, start, dur)
    N = len(data)
    n_batches = N // n_jobs + 1

    print("Extracting mel-spectrogram features")
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
