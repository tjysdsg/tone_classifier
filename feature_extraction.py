import numpy as np
import random
import os
from os.path import join as pjoin
import json
import sys
import argparse
from multiprocessing import Process

data_root = '/NASdata/AudioData/mandarin/AISHELL-2/iOS/data/wav/'
# 声母
INITIALS = ['b', 'c', 'ch', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 'sh', 't', 'w', 'x', 'y',
            'z', 'zh']
OUTDIR = 'feats'
EXISTING_FILES = {t: [] for t in range(5)}


def get_output_path(utt, phone, start, data_dir, postfix=''):
    # `start` is used to distinguish between multiple occurrence of a phone in a sentence
    name = f"{utt}_{phone}_{start:.3f}"
    if len(postfix) > 0:
        name += f'_{postfix}'
    return pjoin(data_dir, f'{name}.npy')


def spectro(y, start: float, dur: float, sr=16000, fmin=50, fmax=350, hop_length=16):
    import librosa

    # mel-spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=2048, hop_length=hop_length, fmin=fmin, fmax=fmax)
    S = librosa.power_to_db(S, ref=np.max)
    # S = S[::-1, :]  # only for visualization

    # crop to the start and the end of a phone
    end = start + dur
    s, e = librosa.time_to_frames([start, end], sr=sr, hop_length=hop_length)
    s = np.max(s, 0)
    e = np.max(e, 0)

    S = S[:, s:e + 1]

    return S


def save_spectro_to_file(S, output_path: str):
    np.save(output_path, S, allow_pickle=False)


def extract_feature(tone: int, utt: str, phone: str, start: float, dur: float):
    import librosa
    spk = utt[1:6]

    outdir = os.path.join(OUTDIR, f'{tone}')
    os.makedirs(outdir, exist_ok=True)
    input_path = pjoin(data_root, spk, f'{utt}.wav')
    existing = EXISTING_FILES[tone]
    y = None

    # original
    outpath = get_output_path(utt, phone, start, outdir)
    if outpath not in existing:
        if y is None:
            y, _ = librosa.load(input_path, sr=16000)

        S = spectro(y, start, dur)
        save_spectro_to_file(S, outpath)
    else:
        sys.stdout.write("\033[K")
        print(f"Skipping {outpath}", end='\r')

    # add random noise
    from aug import add_random_noise, speed_perturb
    outpath_noise = get_output_path(utt, phone, start, outdir, postfix='noise')
    if outpath_noise not in existing:
        if y is None:
            y, _ = librosa.load(input_path, sr=16000)

        snr = random.uniform(50, 60)
        y_noise = add_random_noise(y, snr)
        S = spectro(y_noise, start, dur)
        save_spectro_to_file(S, outpath_noise)
    else:
        sys.stdout.write("\033[K")
        print(f"Skipping {outpath_noise}", end='\r')

    # speed perturb
    outpath_sp09 = get_output_path(utt, phone, start, outdir, postfix='sp09')
    outpath_sp11 = get_output_path(utt, phone, start, outdir, postfix='sp11')
    if outpath_sp09 not in existing:
        if y is None:
            y, _ = librosa.load(input_path, sr=16000)

        y_sp09, y_sp11 = speed_perturb(y)

        S = spectro(y_sp09, start / 0.9, dur / 0.9)
        save_spectro_to_file(S, outpath_sp09)

        S = spectro(y_sp11, start / 1.1, dur / 1.1)
        save_spectro_to_file(S, outpath_sp11)
    else:
        sys.stdout.write("\033[K")
        print(f"Skipping {outpath_sp09} and {outpath_sp11}", end='\r')


def worker(utt2tones: dict, utt):
    tones = utt2tones[utt]
    for t in tones:
        tone, phone, start, dur = t
        extract_feature(tone, utt, phone, start, dur)


def collect_stats():
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
            # phones = [p.replace('5', '0') for p in phones]  # 轻声 5 -> 0
            # separate initials and finals
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
                    utt2trans[utt].append('er5')
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

    all_data = []  # list of (tone, utt, phone, start, dur)
    utt2tones = {}  # {utt: [tone, phone, start, dur]}
    for k, v in utt2time.items():

        trans = utt2trans.get(k)
        if trans is None:
            continue
        if len(trans) != len(v):
            print(f'WARNING: utt {k} different length of transcript and timestamps')
            continue

        for i, p in enumerate(trans):
            tone = p[-1]
            if not tone.isnumeric():  # initials don't have tones, using 0 to represent
                tone = 0
            else:
                tone = int(tone)
                if tone == 5:  # not including light tone
                    continue
            start = v[i][0]
            dur = v[i][1]

            if k not in utt2tones:
                utt2tones[k] = []
            utt2tones[k].append([tone, p, start, dur])
            all_data.append([tone, k, p, start, dur])

    with open('all_data.json', 'w') as f:
        json.dump(all_data, f)
    with open('utt2tones.json', 'w') as f:
        json.dump(utt2tones, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int)
    args = parser.parse_args()

    if args.stage <= 1:
        collect_stats()
    else:
        print("Loading...")

        # find existing files
        # os.path.exists() is slow when the directory contains large amount of files
        for t in range(5):
            folder = os.path.join(OUTDIR, f'{t}')
            os.makedirs(folder, exist_ok=True)
            paths = [d.path for d in os.scandir(folder)]
            EXISTING_FILES[t] = set(paths)

        # load stage 1 output
        n_jobs = 16
        utt2tones: dict = json.load(open('utt2tones.json'))
        utts = list(utt2tones.keys())
        N = len(utts)
        n_batches = N // n_jobs + 1

        print("Extracting mel-spectrogram features")
        for b in range(n_batches):
            print()
            print(f'Batch {b}/{n_batches}')
            offset = b * n_jobs
            ps = [Process(target=worker, args=(utt2tones, utts[offset + i],)) for i in range(n_jobs) if offset + i < N]
            for p in ps:
                p.start()
            for p in ps:
                p.join()


if __name__ == "__main__":
    main()
