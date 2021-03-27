import os
import numpy as np
import json
import argparse
from tqdm import trange
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from train.dataset.dataset import SequentialSpectrogramDataset, collate_sequential_spectrogram
from train.modules.transformers import TransEncoder
from train.modules.models import ContextualModel
from train.utils import (
    set_seed, AverageMeter, masked_accuracy, save_transformer_checkpoint, get_lr, create_logger,
)
import torch
import torch.nn as nn
from train.config import EMBD_DIM, IN_PLANES

NUM_CLASSES = 6  # include initials and light tone, unlike embedding model
SAVE_DIR = 'transformer'
MAX_GRAD_NORM = 10
EMBD_MODEL_EPOCH = 23
os.makedirs(f'exp/{SAVE_DIR}', exist_ok=True)

parser = argparse.ArgumentParser(description='Train embedding on transformer')
parser.add_argument('action', type=str, default='train', nargs='?')
parser.add_argument('--data_dir', default='feats', type=str)
parser.add_argument('--data_name', default='train', type=str)
parser.add_argument('-j', '--workers', default=20, type=int)
parser.add_argument('-b', '--batch_size', default=64, type=int)
parser.add_argument('--val_data_name', default='val', type=str)
parser.add_argument('--train_subset_size', default=0.15, type=float)
parser.add_argument('--test_subset_size', default=0.15, type=float)
parser.add_argument('--val_subset_size', default=0.15, type=float)
parser.add_argument('--embedding_dir', default='embeddings', type=str)
parser.add_argument('--epochs', default=500, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--seed', default=3007123, type=int)
args = parser.parse_args()

set_seed(args.seed)

logger = create_logger('train_transformer', f'exp/{SAVE_DIR}/{args.action}_{args.start_epoch}.log')

utt2tones: dict = json.load(open('utt2tones.json'))


def create_dataloader(utts: list, subset_size: float):
    from sklearn.model_selection import train_test_split
    _, utts = train_test_split(utts, test_size=subset_size, random_state=42)
    u2t = {u: utt2tones[u] for u in utts}

    # count the number of each tone
    tones = {t: 0 for t in range(NUM_CLASSES)}
    for u, t in u2t.items():
        for d in t:
            tone = d[0]
            tones[tone] += 1
    print(tones)

    return DataLoader(
        SequentialSpectrogramDataset(u2t), batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=True, collate_fn=collate_sequential_spectrogram,
    )


# datasets
data_train: list = json.load(open('data_embedding/train_utts.json'))
data_test: list = json.load(open('data_embedding/test_utts.json'))
data_val: list = json.load(open('data_embedding/val_utts.json'))
train_loader = create_dataloader(data_train, args.train_subset_size)
val_loader = create_dataloader(data_val, args.val_subset_size)
test_loader = create_dataloader(data_test, args.test_subset_size)


def load_resnet_embedding_model(model_name: str, epoch: int, in_planes: int, embd_dim: int):
    from train.modules.models import ResNet34StatsPool

    print(f'loading exp/{model_name}/model_{epoch}.pkl')
    ret = ResNet34StatsPool(in_planes, embd_dim).cuda()
    ckpt = torch.load(f'exp/{model_name}/model_{epoch}.pkl')
    ret.load_state_dict(ckpt['model'])
    ret.eval()
    for param in ret.parameters():
        param.requires_grad = False
    return ret


# model, optimizer, criterion, scheduler, trainer
embd_model = load_resnet_embedding_model('embedding', EMBD_MODEL_EPOCH, IN_PLANES, EMBD_DIM).cuda()
trans_encoder = TransEncoder(num_classes=NUM_CLASSES, embedding_size=EMBD_DIM).cuda()
model = ContextualModel(embd_model=embd_model, model=trans_encoder)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum').cuda()
scheduler = ReduceLROnPlateau(optimizer, patience=4, verbose=True)

# load previous model if resume
epochs, start_epoch = args.epochs, args.start_epoch
if start_epoch != 0:
    print(f'Load exp/{SAVE_DIR}/model_{start_epoch - 1}.pkl')
    checkpoint = torch.load(f'exp/{SAVE_DIR}/model_{start_epoch - 1}.pkl')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

model = nn.DataParallel(model)  # must be called after loading


def train():
    for epoch in range(start_epoch, epochs):
        losses, acc = AverageMeter(), AverageMeter()
        model.train()

        # progress bar
        t = trange(len(train_loader))
        t.set_description(f'epoch {epoch}')

        for i, (x, y) in enumerate(train_loader):
            y = y.cuda()
            y_pred, padding_mask = model(x)
            n_samples = y_pred.numel() - torch.sum(padding_mask)

            loss = criterion(torch.flatten(y_pred, 0, 1), torch.flatten(y, 0, 1)) / n_samples

            optimizer.zero_grad()
            loss.backward()
            _ = nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            _ = nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            optimizer.step()

            losses.update(loss.data.item(), n_samples)
            acc.update(masked_accuracy(y_pred, y, padding_mask), n_samples)

            # update progress bar
            t.set_postfix(
                loss=losses.val, loss_avg=float(losses.avg), acc=acc.val, acc_avg=float(acc.avg),
                lr=get_lr(optimizer)
            )
            t.update()

        save_transformer_checkpoint(f'exp/{SAVE_DIR}', epoch, model, optimizer, scheduler)

        acc_val = validate(val_loader)
        logger.info(
            '\nEpoch %d\t  Loss %.4f\t  Accuracy %3.3f\t  lr %f\t  acc_val %3.3f\n'
            % (epoch, float(losses.avg), float(acc.avg), get_lr(optimizer), acc_val)
        )

        scheduler.step(losses.avg)


def validate(dataloader: DataLoader) -> float:
    model.eval()

    acc = AverageMeter()
    ys = []
    preds = []
    with torch.no_grad():
        for j, (packed, y) in enumerate(dataloader):
            items = list(zip(*packed))
            xs = items[0]  # (utts, seq_len', sig_len', mels)
            lengths = items[1]

            y = y.cpu()
            y_pred, padding_mask = model(packed)
            y_pred = y_pred.cpu()
            padding_mask = padding_mask.cpu()

            acc.update(masked_accuracy(y_pred, y, padding_mask), y.size(0))

            for i in range(len(xs)):
                label = y[i, :lengths[i]]
                pred = torch.argmax(y_pred[i][:lengths[i]], dim=-1)

                ys.append(label)
                preds.append(pred)

    ys = torch.cat(ys)
    preds = torch.cat(preds)
    logger.info('Confusion Matrix:')
    confusion = confusion_matrix(ys.numpy(), preds.numpy())
    logger.info(f'\n{confusion}')
    print(acc.avg)

    return acc.avg


if __name__ == '__main__':
    if args.action == 'train':
        train()
    elif args.action == 'test':
        validate(test_loader)
    else:
        raise RuntimeError(f"Unknown action: {args.action}")
