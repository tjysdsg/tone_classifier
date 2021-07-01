import json
import os
import random
from train.dataset.dataset import get_spk_from_utt

SEED = 1024
random.seed(SEED, version=2)

TRAIN_SIZE = 174
OUT_DIR = 'data'
os.makedirs(OUT_DIR, exist_ok=True)


def split(data: list):
    random.shuffle(data)

    train = data[:TRAIN_SIZE]
    test = data[TRAIN_SIZE:]
    return train, test


utt2tones: dict = json.load(open('utt2tones.json'))
all_utts = list(utt2tones.keys())

# speakers
speakers = set()
spk2utts = {}
for u in all_utts:
    spk = get_spk_from_utt(u)
    speakers.add(spk)

    if spk not in spk2utts:
        spk2utts[spk] = []
    spk2utts[spk].append(u)

speakers = list(speakers)
print(f'Number of speakers: {len(speakers)}')  # 1924

train_spks, test_spks = split(speakers)


def get_utts_of_spks(spks: list):
    ret = []
    for spk in spks:
        ret += spk2utts[spk]
    return ret


train_utts = get_utts_of_spks(train_spks)
test_utts = get_utts_of_spks(test_spks)

print(f'Train size: {len(train_utts)}')
print(f'Test size: {len(test_utts)}')

with open(f'{OUT_DIR}/train_utts.json', 'w') as f:
    json.dump(train_utts, f)
with open(f'{OUT_DIR}/test_utts.json', 'w') as f:
    json.dump(test_utts, f)
