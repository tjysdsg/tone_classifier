import json
import os
from feature_extraction import get_output_path


def main():
    utt2tones: dict = json.load(open('utt2tones.json'))
    for utt in utt2tones.keys():
        tones = utt2tones[utt]
        new_tones = []
        for tone, phone, start, dur in tones:
            path = get_output_path(utt, phone, start, f'feats/{tone}')
            print(f'checking {path}', end='\r')
            if os.path.exists(path):
                new_tones.append((tone, phone, start, dur))
        utt2tones[utt] = new_tones
    json.dump(utt2tones, open('utt2tones_fixed.json', 'w'))


if __name__ == '__main__':
    main()
