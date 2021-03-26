import os
import sys
import torch
import logging
from typing import List
import numpy as np
import random


def onehot_encode(idx: int, num_classes: int) -> np.ndarray:
    ret = np.zeros(num_classes)
    ret[idx] = 1
    return ret


def warmup_lr(lr: float, step: int, epoch_size: int, n_warmup_epochs: int):
    return lr / (epoch_size * n_warmup_epochs) * (step + 1)


def get_padding_mask(x: torch.Tensor, lengths: List[int]) -> torch.Tensor:
    batch_size = x.shape[0]
    max_seq_len = x.shape[1]
    padding_mask = torch.zeros(batch_size, max_seq_len + 1, dtype=torch.uint8)
    padding_mask[(torch.arange(batch_size), lengths)] = 1
    padding_mask = padding_mask.cumsum(dim=1)[:, :-1]
    padding_mask = padding_mask > 0  # convert to BoolTensor
    padding_mask = padding_mask
    return padding_mask


def load_transformer_data():
    from sklearn.model_selection import train_test_split
    import json

    utt2tones = json.load(open('utt2tones_fixed.json'))
    utts = list(utt2tones.keys())
    utts_train, utts_test = train_test_split(utts, test_size=0.25)
    utts_train, utts_val = train_test_split(utts_train, test_size=0.1)
    return utt2tones, utts_train, utts_test, utts_val


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_ramdom_state(chk_dir, ramdom_state, np_stats, torch_state, torch_cuda_state):
    torch.save({'random': ramdom_state,
                'np': np_stats,
                'torch': torch_state,
                'torch_cuda': torch_cuda_state
                }, os.path.join(chk_dir, 'random_state.pkl'))


def save_checkpoint(chk_dir, epoch, model, classifier, optimizer, scheduler=None, lr=None):
    torch.save({'model': model.module.state_dict(),
                'classifier': classifier.state_dict() if classifier else None,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'lr': lr
                }, os.path.join(chk_dir, 'model_%d.pkl' % epoch))


def save_transformer_checkpoint(chk_dir, epoch, model, optimizer, scheduler):
    torch.save(
        {
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, os.path.join(chk_dir, 'model_%d.pkl' % epoch)
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target) -> float:
    total = 0
    correct = 0
    _, predicted = torch.max(output.data, 1)
    total += target.size(0)
    correct += (predicted == target).sum().item()
    return correct / total


def masked_accuracy(output, target, padding_mask) -> float:
    total = 0
    correct = 0
    batch_size = output.size(0)
    for i in range(batch_size):
        o = output[i]
        t = target[i]
        mask = padding_mask[i]

        _, pred = torch.max(o.data, 1)
        pred[mask] = 0
        t[mask] = -1

        total += t.size(0) - mask.sum().item()
        correct += (pred == t).sum().item()

    return correct / total


def change_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def create_logger(name: str, log_file: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(fmt='[%(levelname)s] %(asctime)s: %(message)s', datefmt='%Y-%m-%d-%H-%M-%S')

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    return logger


if __name__ == '__main__':
    y_pred = torch.as_tensor(
        [
            [[0.3, 0.7], [0.7, 0.3], [0.3, 0.7]],
            [[0.3, 0.7], [0.3, 0.7], [0.7, 0.3]],
            [[0.3, 0.7], [0.3, 0.7], [0.3, 0.7]]
        ], dtype=torch.float
    )
    y = torch.as_tensor(
        [
            [1, 0, 1],
            [1, 0, 0],
            [1, 0, 0],
        ],
        dtype=torch.long
    )
    mask = torch.as_tensor(
        [
            [True, True, True],
            [True, True, False],
            [True, False, False],
        ],
        dtype=torch.bool
    )

    assert masked_accuracy(y_pred, y, ~mask) == 5 / 6
