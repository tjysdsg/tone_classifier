import os
import argparse
from tqdm import trange
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import json
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from train.dataset.dataset import SequentialEmbeddingDataset, collate_sequential_embedding
from train.modules.transformers import TransEncoder
from train.utils import (set_seed, AverageMeter, masked_accuracy, save_transformer_checkpoint, get_lr)
import torch
import torch.nn as nn

EMBD_DIM = 128
IN_PLANES = 16
NUM_CLASSES = 5

# create output dir
SAVE_DIR = 'transformer'
os.makedirs(f'exp/{SAVE_DIR}', exist_ok=True)

parser = argparse.ArgumentParser(description='Train embedding on transformer')
parser.add_argument('--data_dir', default='feats', type=str)
parser.add_argument('--data_name', default='train', type=str)
parser.add_argument('-j', '--workers', default=20, type=int)
parser.add_argument('-b', '--batch_size', default=64, type=int)
parser.add_argument('--val_data_name', default='val', type=str)
parser.add_argument('--embedding_dir', default='embeddings', type=str)
parser.add_argument('--epochs', default=500, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--seed', default=3007123, type=int)
args = parser.parse_args()

set_seed(args.seed)

utt2tones = json.load(open('utt2tones_fixed.json'))
utts = list(utt2tones.keys())
utts_train, utts_test = train_test_split(utts, test_size=0.25)
utts_train, utts_val = train_test_split(utts_train, test_size=0.1)

# train dataset
train_loader = DataLoader(
    SequentialEmbeddingDataset(utts_train, utt2tones, embedding_dir=args.embedding_dir),
    batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, collate_fn=collate_sequential_embedding,
)

# val dataset
val_loader = DataLoader(
    SequentialEmbeddingDataset(utts_val, utt2tones, embedding_dir=args.embedding_dir),
    batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, collate_fn=collate_sequential_embedding,
)

# model, optimizer, criterion, scheduler, trainer
model = TransEncoder(num_classes=NUM_CLASSES, embedding_size=EMBD_DIM).cuda()
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


def main():
    for epoch in range(start_epoch, epochs):
        losses, acc = AverageMeter(), AverageMeter()
        model.train()

        # progress bar
        t = trange(len(train_loader))
        t.set_description(f'epoch {epoch}')

        for i, (x, y, lengths) in enumerate(train_loader):
            x = x.cuda()
            y = y.cuda()
            y_pred, padding_mask = model(x, lengths)
            n_samples = y_pred.numel() - torch.sum(padding_mask)

            loss = criterion(torch.flatten(y_pred, 0, 1), torch.flatten(y, 0, 1)) / n_samples

            optimizer.zero_grad()
            loss.backward()
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

        acc_val = validate()
        print(
            '\nEpoch %d\t  Loss %.4f\t  Accuracy %3.3f\t  lr %f\t  acc_val %3.3f\n'
            % (epoch, float(losses.avg), float(acc.avg), get_lr(optimizer), acc_val)
        )

        scheduler.step(losses.avg)


def validate() -> float:
    print('=' * 25)
    model.eval()

    acc = AverageMeter()
    with torch.no_grad():
        for j, (x, y, lengths) in enumerate(val_loader):
            y = y.cpu()
            y_pred, padding_mask = model(x, lengths)
            y_pred = y_pred.cpu()
            padding_mask = padding_mask.cpu()

            acc.update(masked_accuracy(y_pred, y, padding_mask), y.size(0))

    return acc.avg


if __name__ == '__main__':
    main()
