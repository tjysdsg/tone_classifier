import json
import os
import random

random.seed(1024, version=2)

# total number of utterances: 619991
TRAIN_SIZE = 100000
TEST_SIZE = 20000
VAL_SIZE = 10000
OUT_DIR = 'data_embedding_6t'

utt2tones: dict = json.load(open('utt2tones.json'))
utts = list(utt2tones.keys())

assert len(utts) >= (TRAIN_SIZE + TEST_SIZE + VAL_SIZE)

random.shuffle(utts)
train_utts = utts[:TRAIN_SIZE]
test_utts = utts[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]
val_utts = utts[TRAIN_SIZE + TEST_SIZE:TRAIN_SIZE + TEST_SIZE + VAL_SIZE]


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


train = flatten_utt2tones(train_utts)
test = flatten_utt2tones(test_utts)
val = flatten_utt2tones(val_utts)

os.makedirs(OUT_DIR, exist_ok=True)
with open(f'{OUT_DIR}/train_utts.json', 'w') as f:
    json.dump(train_utts, f)
with open(f'{OUT_DIR}/test_utts.json', 'w') as f:
    json.dump(test_utts, f)
with open(f'{OUT_DIR}/val_utts.json', 'w') as f:
    json.dump(val_utts, f)

with open(f'{OUT_DIR}/train.json', 'w') as f:
    json.dump(train, f)
with open(f'{OUT_DIR}/test.json', 'w') as f:
    json.dump(test, f)
with open(f'{OUT_DIR}/val.json', 'w') as f:
    json.dump(val, f)
