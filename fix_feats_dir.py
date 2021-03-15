"""
Create files required by train_embedding.py:

- wav.scp
- utt2spk
- spk2utt

To avoid renaming all the files and strings, 'speaker' means tone index (0 = 1st tone, 1 = 2nd tone, etc).
"""

import os

DATA_DIR = 'feats'
TONES = list(range(4))


def main():
    wavscp = []
    utt2spk = []
    spk2utt = []

    for t in TONES:
        for f in os.scandir(os.path.join(DATA_DIR, str(t))):
            path = f.path

            filename = f.name
            tokens = filename.split('_')
            utt, phone, start = tokens[:3]

            wavscp.append(f'{filename}\t{path}\n')
            utt2spk.append(f'{filename}\t{t}\n')
            spk2utt.append(f'{t}\t{filename}\n')

    with open(os.path.join(DATA_DIR, 'wav.scp'), 'w') as f:
        f.writelines(wavscp)
    with open(os.path.join(DATA_DIR, 'utt2spk'), 'w') as f:
        f.writelines(utt2spk)
    with open(os.path.join(DATA_DIR, 'spk2utt'), 'w') as f:
        f.writelines(spk2utt)


if __name__ == '__main__':
    main()
