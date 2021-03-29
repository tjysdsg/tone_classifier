import argparse
import json
from tqdm import trange
import numpy as np
from train.utils import (
    set_seed, create_logger, AverageMeter, accuracy, save_checkpoint, get_lr,
)
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train.modules.models import *
from train.dataset.dataset import collate_spectrogram, PhoneSegmentDataset
from train.config import NUM_CLASSES, EMBD_DIM, IN_PLANES

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='Training embedding')
parser.add_argument('--save_dir', type=str)

parser.add_argument('-j', '--workers', default=8, type=int)
parser.add_argument('-b', '--batch_size', default=64, type=int)

parser.add_argument('--data_dir', default='data', type=str)
parser.add_argument('--val_data_name', default='val', type=str)
parser.add_argument('--test_data_name', default='test', type=str)
parser.add_argument('--train_subset_size', default=0.02, type=float)
parser.add_argument('--test_subset_size', default=0.02, type=float)
parser.add_argument('--val_subset_size', default=0.02, type=float)

parser.add_argument('--include_dur', default=False, action='store_true')
parser.add_argument('--include_onehot', default=False, action='store_true')
parser.add_argument('--include_context', default=False, action='store_true')

parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--warm_up_epoch', default=3, type=int)
parser.add_argument('--lr_patience', default=2, type=int)

parser.add_argument('action', type=str, default='train', nargs='?')
parser.add_argument('--epochs', default=500, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--seed', default=3007123, type=int)
args = parser.parse_args()
set_seed(args.seed)

DATA_DIR = args.data_dir
SAVE_DIR = args.save_dir

os.makedirs(f'exp/{SAVE_DIR}', exist_ok=True)
print(f'Saving logs and output to exp/{SAVE_DIR}')

INCLUDE_DUR = args.include_dur
INCLUDE_ONEHOT = args.include_onehot
INCLUDE_CONTEXT = args.include_context
print(f'Using durations: {INCLUDE_DUR}\nUsing onehot encodings: {INCLUDE_ONEHOT}\nUsing context: {INCLUDE_CONTEXT}')

logger = create_logger('train_embedding', f'exp/{SAVE_DIR}/{args.action}_{args.start_epoch}.log')

utt2tones: dict = json.load(open('utt2tones.json'))


def create_dataloader(utts: list, subset_size: float):
    from sklearn.model_selection import train_test_split
    _, utts = train_test_split(utts, test_size=subset_size, random_state=42)
    u2t = {u: utt2tones[u] for u in utts}

    # count the number of each tone
    tones = {t: 0 for t in range(NUM_CLASSES)}
    for _, t in u2t.items():
        for d in t:
            tone = d[0]
            tones[tone] += 1
    print(tones)

    return DataLoader(
        PhoneSegmentDataset(
            u2t, feat_type='spectrogram', include_dur=INCLUDE_DUR, include_onehot=INCLUDE_ONEHOT,
            include_context=INCLUDE_CONTEXT,
        ),
        batch_size=args.batch_size, num_workers=args.workers, collate_fn=collate_spectrogram,
    )


# data loaders
data_train: list = json.load(open(f'{DATA_DIR}/train_utts.json'))
data_test: list = json.load(open(f'{DATA_DIR}/test_utts.json'))
data_val: list = json.load(open(f'{DATA_DIR}/val_utts.json'))
train_loader = create_dataloader(data_train, args.train_subset_size)
val_loader = create_dataloader(data_val, args.val_subset_size)
test_loader = create_dataloader(data_test, args.test_subset_size)

print('train size:', len(train_loader) * args.batch_size)
print('test size:', len(test_loader) * args.batch_size)
print('val size:', len(val_loader) * args.batch_size)

# models
inner_model = ResNet34StatsPool(IN_PLANES, EMBD_DIM, dropout=0.5).cuda()
# TDNNStatsPool(embedding_size=EMBD_DIM).cuda()
# BLSTMStatsPool(embedding_size=EMBD_DIM).cuda()
model = EmbeddingModel(
    inner_model, EMBD_DIM, NUM_CLASSES, include_dur=INCLUDE_DUR,
    include_onehot=INCLUDE_ONEHOT, include_context=INCLUDE_CONTEXT,
).cuda()

# criterion, optimizer, scheduler
criterion = nn.CrossEntropyLoss().cuda()
lr = args.lr
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.lr_patience, factor=0.1)

# load previous model if resume
epochs, start_epoch = args.epochs, args.start_epoch
if start_epoch != 0:
    print(f'Load exp/{SAVE_DIR}/model_{start_epoch - 1}.pkl')
    checkpoint = torch.load(f'exp/{SAVE_DIR}/model_{start_epoch - 1}.pkl')
    model.load_state_dict(checkpoint['model'])
    if not args.lr:
        optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

model = nn.DataParallel(model)


def train():
    for epoch in range(start_epoch, epochs):
        losses, acc = AverageMeter(), AverageMeter()
        model.train()

        # progress bar
        t = trange(len(train_loader))
        t.set_description(f'epoch {epoch}')

        for i, packed in enumerate(train_loader):
            x = packed[0]
            y = packed[1]

            durs = None
            onehots = None
            if len(packed) >= 3:
                durs = packed[2]

            if len(packed) >= 4:
                onehots = packed[3]

            x, y = x.cuda(), y.cuda()

            y_pred = model(x, durs, onehots)
            loss = criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.data.item(), x.size(0))
            acc.update(accuracy(y_pred.data, y), x.size(0))

            # update progress bar
            t.set_postfix(loss=losses.val, loss_avg=losses.avg, acc=acc.val, acc_avg=acc.avg, lr=get_lr(optimizer))
            t.update()

        save_checkpoint(f'exp/{SAVE_DIR}', epoch, model, optimizer, scheduler)

        acc_val = validate(val_loader)
        logger.info(
            '\nEpoch %d\t  Loss %.4f\t  Accuracy %3.3f\t  lr %f\t  acc_val %3.3f\n'
            % (epoch, losses.avg, acc.avg, get_lr(optimizer), acc_val)
        )

        scheduler.step(losses.avg)


def validate(dataloader: DataLoader) -> float:
    model.eval()

    ys = []
    preds = []
    with torch.no_grad():
        for j, packed in enumerate(dataloader):
            x = packed[0]
            y = packed[1]

            durs = None
            onehots = None
            if len(packed) >= 3:
                durs = packed[2]

            if len(packed) >= 4:
                onehots = packed[3]

            y_pred = model(x.cuda(), durs, onehots).cpu()
            y = y.cpu()
            ys.append(y)
            preds.append(torch.argmax(y_pred, dim=-1))

    ys = torch.cat(ys).numpy()
    preds = torch.cat(preds).numpy()

    acc = accuracy_score(ys, preds)
    logger.info(f'Accuracy: {acc}')

    confusion = confusion_matrix(ys, preds)
    logger.info(f'\n{confusion}')

    report = classification_report(ys, preds)
    logger.info(f'\n{report}')
    return acc


if __name__ == '__main__':
    if args.action == 'train':
        train()
    elif args.action == 'test':
        validate(test_loader)
    else:
        raise RuntimeError(f"Unknown action: {args.action}")
