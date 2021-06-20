import json
import os
from train.config import WAV_DIR
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--utts', type=str)
args = parser.parse_args()


def get_audio_duration(path: str):
    import librosa
    return librosa.get_duration(filename=path)


def main():
    utts = json.load(open(args.utts))

    utt2dur = {}
    for spk in os.scandir(WAV_DIR):
        if spk.is_dir() and spk.name.startswith('SSB'):
            for file in os.scandir(spk.path):
                if file.is_file():
                    utt = file.name.replace('.wav', '')
                    utt2dur[utt] = get_audio_duration(file.path)

    res = 0
    for utt, dur in utt2dur.items():
        if utt in utts:
            res += dur

    json.dump(res, open('durations.json', 'w'))

    print(res / 3600)


if __name__ == '__main__':
    main()
