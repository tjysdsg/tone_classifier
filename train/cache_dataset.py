import argparse
import json
from tqdm import trange
from train.dataset.dataset import CachedSpectrogramExtractor
from multiprocessing import Process, Manager

parser = argparse.ArgumentParser(description='Make cache of training data')
parser.add_argument('-j', '--workers', default=20, type=int)
parser.add_argument('--data_dir', default='data', type=str)
args = parser.parse_args()
DATA_DIR = args.data_dir

# data loaders
data_train: list = json.load(open(f'{DATA_DIR}/train_utts.json'))
data_test: list = json.load(open(f'{DATA_DIR}/test_utts.json'))
utts = data_train + data_test

# TODO: pass in shared dictionary to extractor slows down the program
extractor = CachedSpectrogramExtractor()


def main():
    n_jobs = args.workers
    N = len(utts)
    n_batches = N // n_jobs + 1

    t = trange(n_batches)
    for b in range(n_batches):
        offset = b * n_jobs
        ps = [
            Process(target=extractor.load_utt, args=(utts[offset + i],))
            for i in range(n_jobs)
            if offset + i < N
        ]
        for p in ps:
            p.start()
        for p in ps:
            p.join()

        t.update()


if __name__ == '__main__':
    main()
