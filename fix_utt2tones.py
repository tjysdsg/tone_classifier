import json
import os
from feature_extraction import get_output_path


def main():
    utt2tones: dict = json.load(open('utt2tones.json'))
    utt2tones_fixed = {}
    for utt, tones in utt2tones.items():
        new_tones = []
        for tone, phone, start, dur in tones:
            path = get_output_path(utt, phone, start, f'feats/{tone}')
            print(f'checking {path}', end='\r')
            if os.path.exists(path):
                new_tones.append((tone, phone, start, dur))

        if len(new_tones) > 0:
            utt2tones_fixed[utt] = new_tones

    json.dump(utt2tones_fixed, open('utt2tones_fixed.json', 'w'))


if __name__ == '__main__':
    main()
