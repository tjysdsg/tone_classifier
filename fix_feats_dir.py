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
        data = []
        for f in os.scandir(os.path.join(DATA_DIR, str(t))):
            path = f.path
            filename = f.name
            data.append([filename, path])

        from sklearn.model_selection import train_test_split
        data_train, data_test = train_test_split(data, test_size=0.25)
        data_train, data_val = train_test_split(data_train, test_size=0.1)

        data = dict(train=data_train, test=data_test, val=data_val)
        for subset, d in data.items():
            for e in d:
                filename, path = e
                wavscp.append(f'{filename}\t{path}\n')
                utt2spk.append(f'{filename}\t{t}\n')
                spk2utt.append(f'{t}\t{filename}\n')

            subdir = os.path.join(DATA_DIR, subset)
            os.makedirs(subdir, exist_ok=True)
            with open(os.path.join(subdir, 'wav.scp'), 'w') as f:
                f.writelines(wavscp)
            with open(os.path.join(subdir, 'utt2spk'), 'w') as f:
                f.writelines(utt2spk)
            with open(os.path.join(subdir, 'spk2utt'), 'w') as f:
                f.writelines(spk2utt)


if __name__ == '__main__':
    main()
