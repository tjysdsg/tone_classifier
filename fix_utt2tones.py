import json
import os
from feature_extraction import get_output_path


def main():
    existing_files = []
    for t in range(5):
        folder = os.path.join('feats', f'{t}')
        os.makedirs(folder, exist_ok=True)
        paths = [d.path for d in os.scandir(folder)]
        existing_files += paths
    existing_files = set(existing_files)

    utt2tones: dict = json.load(open('utt2tones.json'))
    utt2tones_fixed = {}
    for utt, tones in utt2tones.items():
        new_tones = []
        for tone, phone, start, dur in tones:
            path = get_output_path(utt, phone, start, f'feats/{tone}')
            print(f'checking {path}', end='\r')
            if path in existing_files:
                new_tones.append((tone, phone, start, dur))

        if len(new_tones) > 0:
            utt2tones_fixed[utt] = new_tones

    json.dump(utt2tones_fixed, open('utt2tones_fixed.json', 'w'))


if __name__ == '__main__':
    main()
