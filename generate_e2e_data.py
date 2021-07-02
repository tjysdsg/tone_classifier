import os
import json
import argparse

WAV_DIR = '/NASdata/AudioData/AISHELL-ASR-SSB/SPEECHDATA/'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--utts', type=str)
    parser.add_argument('--utt2tones', type=str)
    parser.add_argument('--output-dir', type=str)
    return parser.parse_args()


def get_spk_from_utt(utt: str):
    return utt[:7]


def get_wav_path(utt: str):
    spk = get_spk_from_utt(utt)
    return os.path.join(WAV_DIR, spk, f'{utt}.wav')


def main():
    args = get_args()

    utts = json.load(open(args.utts))
    utts = set(utts)
    utt2data = json.load(open(args.utt2tones))

    utt2tones = {}
    for utt, data in utt2data.items():
        if utt not in utts:
            continue

        tones = [str(d[0]) for d in data]  # [tone, phone, start, dur]
        utt2tones[utt] = tones

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'text'), 'w') as f:
        for utt, tones in utt2tones.items():
            t = ' '.join(tones)
            f.write(f'{utt}\t{t}\n')

    with open(os.path.join(output_dir, 'wav.scp'), 'w') as f:
        for utt, _ in utt2tones.items():
            p = get_wav_path(utt)
            f.write(f'{utt}\t{p}\n')


if __name__ == '__main__':
    main()
