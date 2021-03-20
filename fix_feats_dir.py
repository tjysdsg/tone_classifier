"""
Create files required by train_embedding.py:

- wav.scp
- utt2spk
- spk2utt

To avoid renaming all the files and strings, 'speaker' means tone index (0 = 1st tone, 1 = 2nd tone, etc).
"""

import os
import numpy as np

DATA_DIR = 'feats'
TONES = list(range(5))


def main():
    tone_data = {t: [] for t in TONES}
    visited = set()
    for t in TONES:
        for f in os.scandir(os.path.join(DATA_DIR, str(t))):
            path = f.path
            filename = f.name

            if 'noise' in filename or 'sp' in filename:
                # remove '_noise' and '_sp09'/'_sp11' suffix
                filename, ext = os.path.splitext(filename)
                orig_name = '_'.join(filename.split('_')[:-1]) + ext

                print(f'{orig_name} is augmented', end='\r')

                # only include data that is augmented
                if orig_name not in visited:
                    tone_data[t].append([orig_name, path])
                    visited.add(orig_name)

    min_len = np.min([len(v) for k, v in tone_data.items()])
    print(f'Balanced data size of each tone is {min_len}')

    data = []
    for t in TONES:
        for filename, path in tone_data[t]:
            data.append([filename, path, t])

    from sklearn.model_selection import train_test_split
    data_train, data_test = train_test_split(data, shuffle=True, test_size=0.25)
    data_train, data_val = train_test_split(data_train, shuffle=True, test_size=0.1)

    data = dict(train=data_train, test=data_test, val=data_val)
    for subset, d in data.items():
        wavscp = []
        utt2spk = []
        spk2utt = []

        for e in d:
            filename, path, tone = e
            wavscp.append(f'{filename}\t{path}\n')
            utt2spk.append(f'{filename}\t{tone}\n')
            spk2utt.append(f'{tone}\t{filename}\n')

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
