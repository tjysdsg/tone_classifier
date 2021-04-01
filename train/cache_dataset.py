import argparse
import json
from tqdm import trange
from train.utils import set_seed
import torch
from train.dataset.dataset import create_dataloader

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='Make cache of training data')
parser.add_argument('-j', '--workers', default=8, type=int)
parser.add_argument('-b', '--batch_size', default=64, type=int)
parser.add_argument('--data_dir', default='data', type=str)
parser.add_argument('--subset_size', default=0.05, type=float)
parser.add_argument('--seed', default=3007123, type=int)
args = parser.parse_args()
set_seed(args.seed)
DATA_DIR = args.data_dir

utt2tones: dict = json.load(open('utt2tones.json'))

# data loaders
data_train: list = json.load(open(f'{DATA_DIR}/train_utts.json'))
data_test: list = json.load(open(f'{DATA_DIR}/test_utts.json'))
data_val: list = json.load(open(f'{DATA_DIR}/val_utts.json'))
all_data = data_train + data_test + data_val
loader = create_dataloader(
    all_data, utt2tones, args.subset_size, batch_size=args.batch_size, n_workers=args.workers
)


def main():
    # progress bar
    t = trange(len(loader))

    for _, _ in enumerate(loader):
        t.update()


if __name__ == '__main__':
    main()
