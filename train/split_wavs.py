import json
import os
import random
from sklearn.model_selection import train_test_split
from train.dataset.dataset import get_spk_from_utt

SEED = 1024
random.seed(SEED, version=2)

TRAIN_SIZE = 0.8
TEST_SIZE = 0.1
VAL_SIZE = 0.1
OUT_DIR = 'data/embedding_6t'
os.makedirs(OUT_DIR, exist_ok=True)


def split(data: list):
    total = len(data)
    random.shuffle(data)

    train_size = int(total * TRAIN_SIZE)
    test_size = int(total * TEST_SIZE)
    val_size = int(total * VAL_SIZE)

    train = data[:train_size]
    test = data[train_size:train_size + test_size]
    val = data[train_size + test_size:train_size + test_size + val_size]
    return train, test, val


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
print(f'Number of speakers: {len(speakers)}')

train_utts = []
test_utts = []
val_utts = []
ivector_utts = []

for spk, us in spk2utts.items():
    if len(us) < 100:
        print(f'Speaker {spk} contains too few utterances {len(us)}')
        continue

    tr, t, val = split(us)
    tr, iv = train_test_split(tr, test_size=0.2, random_state=SEED)
    train_utts += tr
    test_utts += t
    val_utts += val
    ivector_utts += iv

print(f'Train size: {len(train_utts)}')
print(f'Test size: {len(test_utts)}')
print(f'Val size: {len(val_utts)}')
print(f'i-vector size: {len(ivector_utts)}')

with open(f'{OUT_DIR}/train_utts.json', 'w') as f:
    json.dump(train_utts, f)
with open(f'{OUT_DIR}/test_utts.json', 'w') as f:
    json.dump(test_utts, f)
with open(f'{OUT_DIR}/val_utts.json', 'w') as f:
    json.dump(val_utts, f)
with open(f'{OUT_DIR}/ivector_utts.json', 'w') as f:
    json.dump(ivector_utts, f)


def flatten_utt2tones(utts: list):
    ret = []
    for utt in utts:
        for tone, phone, start, dur in utt2tones[utt]:
            # if tone != 5 and tone != 0:  # not including initials or light tone
            #     tone -= 1  # tone label starts at 0
            #     ret.append([tone, utt, phone, start, dur])
            ret.append([tone, utt, phone, start, dur])
    random.shuffle(ret)
    return ret

# train = flatten_utt2tones(train_utts)
# test = flatten_utt2tones(test_utts)
# val = flatten_utt2tones(val_utts)
# with open(f'{OUT_DIR}/train.json', 'w') as f:
#     json.dump(train, f)
# with open(f'{OUT_DIR}/test.json', 'w') as f:
#     json.dump(test, f)
# with open(f'{OUT_DIR}/val.json', 'w') as f:
#     json.dump(val, f)
