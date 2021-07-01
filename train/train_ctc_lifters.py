import argparse
import sys
import numpy as np
from typing import Tuple
from tqdm import trange
from train.utils import (
    set_seed, create_logger, AverageMeter, accuracy, save_checkpoint, get_lr,
)
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train.modules.end2end import End2End
from train.dataset.dataset import CepstrumDataset, collate_cepstrum
from train.config import NUM_CLASSES, EMBD_DIM, IN_PLANES, MAX_GRAD_NORM

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='Training embedding')
parser.add_argument('--save-dir', type=str)

parser.add_argument('-j', '--workers', default=10, type=int)
parser.add_argument('-b', '--batch_size', default=32, type=int)

parser.add_argument('--train-dir', type=str)
parser.add_argument('--test-dir', type=str)

parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--lr-patience', default=1, type=int)

parser.add_argument('action', type=str, default='train', nargs='?')
parser.add_argument('--epochs', default=500, type=int)
parser.add_argument('--start-epoch', default=0, type=int)
parser.add_argument('--seed', default=3007123, type=int)
args = parser.parse_args()
set_seed(args.seed)

SAVE_DIR = args.save_dir

os.makedirs(f'exp/{SAVE_DIR}', exist_ok=True)
print(f'Saving logs and output to exp/{SAVE_DIR}')

logger = create_logger('train ctc with lifters', f'exp/{SAVE_DIR}/{args.action}_{args.start_epoch}.log')
logger.info(" ".join(sys.argv))  # save entire command for reproduction

VOCAB_SIZE = NUM_CLASSES + 1  # num classes + <blank>


def _create_loader(text: str, wavscp: str):
    return DataLoader(
        CepstrumDataset(text, wavscp),
        batch_size=args.batch_size, num_workers=args.workers, collate_fn=collate_cepstrum,
    )


train_loader = _create_loader(os.path.join(args.train_dir, 'text'), os.path.join(args.train_dir, 'wav.scp'))
test_loader = _create_loader(os.path.join(args.test_dir, 'text'), os.path.join(args.test_dir, 'wav.scp'))
print('train size:', len(train_loader) * args.batch_size)
print('test size:', len(test_loader) * args.batch_size)

# models
model = End2End(128, VOCAB_SIZE).cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
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


def step(batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x, y, y_lengths = batch
    y_pred, pred_lengths = model(x.cuda())
    return x, y, y_pred, pred_lengths, y_lengths


def train():
    for epoch in range(start_epoch, epochs):
        losses = AverageMeter()
        model.train()

        # progress bar
        t = trange(len(train_loader))
        t.set_description(f'epoch {epoch}')

        for i, batch in enumerate(train_loader):
            x, y, y_pred, pred_lengths, y_lengths = step(batch)
            loss = model.module.loss(y_pred, y, pred_lengths, y_lengths)

            optimizer.zero_grad()
            loss.backward()

            _ = nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            _ = nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            optimizer.step()

            losses.update(loss.data.item(), y.size(0))

            # update progress bar
            t.set_postfix(loss=losses.val, loss_avg=losses.avg, lr=get_lr(optimizer))
            t.update()

        save_checkpoint(f'exp/{SAVE_DIR}', epoch, model, optimizer, scheduler)

        ter = validate(test_loader)
        logger.info(
            '\nEpoch %d\tLoss %.4f\tlr %f\tTER %3.3f\n'
            % (epoch, losses.avg, get_lr(optimizer), ter)
        )

        scheduler.step(losses.avg)


def infer(dataloader: DataLoader) -> Tuple[list, list]:
    model.eval()

    ys = []
    preds = []
    with torch.no_grad():
        for j, batch in enumerate(dataloader):
            x, y, y_lengths = batch

            y = y.detach().cpu().numpy().tolist()
            labels = [tones[:y_lengths[i]] for i, tones in enumerate(y)]

            y_pred = model.module.predict(x.cuda())

            ys += labels
            preds += y_pred

    return ys, preds


def validate(dataloader: DataLoader) -> float:
    from train.wer import wer_details_for_batch, wer_summary

    ys, preds = infer(dataloader)
    assert len(ys) == len(preds)

    utts = [f'{i}' for i in range(len(ys))]
    res = wer_details_for_batch(utts, ys, preds)
    ter = wer_summary(res)['WER']

    logger.info(f'TER: {ter}')
    return ter


if __name__ == '__main__':
    if args.action == 'train':
        model = nn.DataParallel(model)
        train()
    elif args.action == 'test':
        model = nn.DataParallel(model)
        validate(test_loader)
    else:
        raise RuntimeError(f"Unknown action: {args.action}")
