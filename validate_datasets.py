"""Make sure there is no overlapping speakers in train, test, val set"""

import json
import os
from train.dataset.dataset import get_spk_from_utt

DATA_DIR = 'data'


def main():
    subsets = ['train', 'test', 'val']
    sets = {s: [] for s in subsets}
    for x in subsets:
        file = os.path.join(DATA_DIR, f'{x}_utts.json')
        utts = json.load(open(file))

        for utt in utts:
            spk = get_spk_from_utt(utt)
            sets[x].append(spk)

    sets = {s: set(utts) for s, utts in sets.items()}
    for s1, utts1 in sets.items():  # type: (str, set)
        for s2, utts2 in sets.items():  # type: (str, set)
            if s1 == s2:
                continue
            intersection = utts1.intersection(utts2)
            if len(intersection) > 0:
                print(f'Subset {s1} has overlapping speakers with subset {s2}\n{intersection}')


if __name__ == '__main__':
    main()
