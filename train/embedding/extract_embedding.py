import torch
from tqdm import tqdm
import numpy as np
import os
from torch.utils.data import DataLoader
from train.dataset.dataset import SequentialSpectrogramDataset, collate_sequential_spectorgram
from train.modules.models import ResNet34StatsPool
import json
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')  # https://github.com/pytorch/pytorch/issues/11201

EMBD_DIM = 128
IN_PLANES = 16
BATCH_SIZE = 64
EMBD_MODEL_EPOCH = 23
OUT_DIR = 'exp/extract_embeddings/'
os.makedirs(OUT_DIR, exist_ok=True)


def load_resnet_embedding_model(model_name: str, epoch: int, in_planes: int, embd_dim: int) -> ResNet34StatsPool:
    print(f'loading exp/{model_name}/model_{epoch}.pkl')
    model = ResNet34StatsPool(in_planes, embd_dim).cuda()
    checkpoint = torch.load(f'exp/{model_name}/model_{epoch}.pkl')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


embd_model = load_resnet_embedding_model('embedding', EMBD_MODEL_EPOCH, IN_PLANES, EMBD_DIM).cuda()

utt2tones: dict = json.load(open('utt2tones.json'))
utts: list = json.load(open('data_embedding/train_utts.json'))
utt2tones = {u: utt2tones[u] for u in utts}
print('Size:', len(utt2tones))
loader = DataLoader(
    SequentialSpectrogramDataset(utt2tones), batch_size=BATCH_SIZE, num_workers=20, pin_memory=True,
    collate_fn=collate_sequential_spectorgram,
)


def main():
    t = tqdm(iterable=False, total=float('inf'))
    for i, (utts, xs, ys) in enumerate(loader):
        n = len(xs)
        for j in range(n):
            utt = utts[j]

            output_path = os.path.join(OUT_DIR, f'{utt}.npy')
            if os.path.exists(output_path):
                t.set_postfix(skipped=utt)
                t.update()
                continue

            x = xs[j]  # (seq_len, n_frames, mels)
            # y = ys[j]  # (seq_len, )

            x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True).cuda()
            embedding = embd_model(x).cpu().numpy()
            np.save(output_path, embedding)

            t.set_postfix(now=utt)
            t.update()


if __name__ == '__main__':
    main()
