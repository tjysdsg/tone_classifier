import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from train.dataset.dataset import SequentialSpectrogramDataset, collate_sequential_spectorgram
from train.utils import load_embedding_model
import json

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')  # https://github.com/pytorch/pytorch/issues/11201

EMBD_DIM = 128
IN_PLANES = 16
BATCH_SIZE = 64
EMBD_MODEL_EPOCH = 7
OUT_DIR = 'embeddings'

embd_model = load_embedding_model(EMBD_MODEL_EPOCH, IN_PLANES, EMBD_DIM).cuda()

with open('utt2tones_fixed.json') as f:
    utt2tones = json.load(f)

utts = list(utt2tones.keys())
loader = DataLoader(
    SequentialSpectrogramDataset(utts, utt2tones), batch_size=BATCH_SIZE, num_workers=20, pin_memory=True,
    collate_fn=collate_sequential_spectorgram,
)

os.makedirs(OUT_DIR, exist_ok=True)


def main():
    for i, (utts, xs, ys) in enumerate(loader):
        n = len(xs)
        for j in range(n):
            utt = utts[j]

            output_path = os.path.join(OUT_DIR, f'{utt}.npy')
            if os.path.exists(output_path):
                print(f'Skipping {utt}')
                continue

            print(f'Calculating embedding for {utt}')

            x = xs[j]  # (seq_len, n_frames, mels)
            # y = ys[j]  # (seq_len, )

            x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True).cuda()
            embedding = embd_model(x).cpu().numpy()
            np.save(output_path, embedding)


if __name__ == '__main__':
    main()
