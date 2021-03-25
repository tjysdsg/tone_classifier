import argparse
import json
from tqdm import trange
import numpy as np
from train.utils import (
    set_seed, create_logger, AverageMeter, accuracy, save_checkpoint, save_ramdom_state, get_lr,
)
from sklearn.metrics import confusion_matrix, accuracy_score
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train.modules.models import ResNet34StatsPool, TDNNStatsPool, BLSTMStatsPool
from train.dataset.dataset import SpectrogramDataset, collate_fn_pad
from train.config import NUM_CLASSES, EMBD_DIM, IN_PLANES

# create output dir
SAVE_DIR = 'embedding_aug'
os.makedirs(f'exp/{SAVE_DIR}', exist_ok=True)

parser = argparse.ArgumentParser(description='Training embedding')
# dataset
parser.add_argument('--data_dir', default='feats', type=str)
parser.add_argument('--data_name', default='train', type=str)
parser.add_argument('-j', '--workers', default=8, type=int)
parser.add_argument('-b', '--batch_size', default=64, type=int)
parser.add_argument('--val_data_name', default='val', type=str)
parser.add_argument('--test_data_name', default='test', type=str)
parser.add_argument('--train_subset_size', default=0.03, type=float)
parser.add_argument('--test_subset_size', default=0.1, type=float)
parser.add_argument('--val_subset_size', default=0.1, type=float)
# learning rate scheduler
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--warm_up_epoch', default=3, type=int)
parser.add_argument('--lr_patience', default=4, type=int)
# others
parser.add_argument('action', type=str, default='train', nargs='?')
parser.add_argument('--epochs', default=500, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--seed', default=3007123, type=int)
args = parser.parse_args()

set_seed(args.seed)

logger = create_logger('train_embedding', f'exp/{SAVE_DIR}/{args.action}_{args.start_epoch}.log')


def create_dataloader(data: list, subset_size: float):
    from sklearn.model_selection import train_test_split
    _, data = train_test_split(data, test_size=subset_size)

    # count the number of each tone
    tones = {t: 0 for t in range(NUM_CLASSES)}
    for d in data:
        tone = d[0]
        tones[tone] += 1
    print(tones)

    return DataLoader(
        SpectrogramDataset(data), batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=True, collate_fn=collate_fn_pad,
    )


# data loaders
data_train: list = json.load(open('data_embedding/train.json'))
data_test: list = json.load(open('data_embedding/test.json'))
data_val: list = json.load(open('data_embedding/val.json'))
train_loader = create_dataloader(data_train, args.train_subset_size)
val_loader = create_dataloader(data_val, args.val_subset_size)
test_loader = create_dataloader(data_test, args.test_subset_size)

print('train size:', len(train_loader) * args.batch_size)
print('test size:', len(test_loader) * args.batch_size)
print('val size:', len(val_loader) * args.batch_size)

# models
# model = ResNet34StatsPool(IN_PLANES, EMBD_DIM, dropout=0.5).cuda()
model = TDNNStatsPool(embedding_size=EMBD_DIM).cuda()
# model = BLSTMStatsPool(embedding_size=EMBD_DIM).cuda()
classifier = nn.Linear(EMBD_DIM, NUM_CLASSES).cuda()

# criterion, optimizer, scheduler
criterion = nn.CrossEntropyLoss().cuda()
lr = args.lr
optimizer = torch.optim.SGD(
    list(model.parameters()) + list(classifier.parameters()), lr=lr, momentum=0.9,
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.lr_patience, factor=0.1)

# load previous model if resume
epochs, start_epoch = args.epochs, args.start_epoch
if start_epoch != 0:
    print(f'Load exp/{SAVE_DIR}/model_{start_epoch - 1}.pkl')
    checkpoint = torch.load(f'exp/{SAVE_DIR}/model_{start_epoch - 1}.pkl')
    model.load_state_dict(checkpoint['model'])
    classifier.load_state_dict(checkpoint['classifier'])
    if not args.lr:
        optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    checkpoint = torch.load(f'exp/{SAVE_DIR}/random_state.pkl')
    random.setstate(checkpoint['random'])
    np.random.set_state(checkpoint['np'])
    torch.set_rng_state(checkpoint['torch'])

model = nn.DataParallel(model)


def train():
    for epoch in range(start_epoch, epochs):
        losses, acc = AverageMeter(), AverageMeter()
        model.train()
        classifier.train()

        # progress bar
        t = trange(len(train_loader))
        t.set_description(f'epoch {epoch}')

        for i, (feats, label) in enumerate(train_loader):
            # if epoch < args.warm_up_epoch:
            #     change_lr(
            #         optimizer,
            #         warmup_lr(lr, len(train_loader) * epoch + i, len(train_loader), args.warm_up_epoch)
            #     )

            feats, label = feats.cuda(), label.cuda()

            outputs = classifier(model(feats))
            loss = criterion(outputs, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.data.item(), feats.size(0))
            acc.update(accuracy(outputs.data, label), feats.size(0))

            # update progress bar
            t.set_postfix(loss=losses.val, loss_avg=losses.avg, acc=acc.val, acc_avg=acc.avg,
                          lr=get_lr(optimizer))
            t.update()

        save_checkpoint(f'exp/{SAVE_DIR}', epoch, model, classifier, optimizer, scheduler)
        # noinspection PyUnresolvedReferences
        save_ramdom_state(
            f'exp/{SAVE_DIR}', random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all()
        )

        acc_val = validate()
        logger.info(
            '\nEpoch %d\t  Loss %.4f\t  Accuracy %3.3f\t  lr %f\t  acc_val %3.3f\n'
            % (epoch, losses.avg, acc.avg, get_lr(optimizer), acc_val)
        )

        scheduler.step(losses.avg)


def validate() -> float:
    logger.info('============== VALIDATING ==============')
    model.eval()
    classifier.eval()

    ys = []
    preds = []
    with torch.no_grad():
        for j, (x, y) in enumerate(val_loader):
            y = y.cpu()
            y_pred = classifier(model(x)).cpu()
            ys.append(y)
            preds.append(torch.argmax(y_pred, dim=-1))

    ys = torch.cat(ys)
    preds = torch.cat(preds)

    confusion = confusion_matrix(ys.numpy(), preds.numpy())
    logger.info('Confusion Matrix:')
    logger.info(confusion)

    # confusion = confusion[1:, 1:]
    # logger.info(f'4 tone accuracy: {np.trace(confusion) / np.sum(confusion)}')

    return accuracy_score(ys.numpy(), preds.numpy())


def test():
    logger.info('============== TESTING ==============')
    model.eval()
    classifier.eval()

    ys = []
    preds = []
    with torch.no_grad():
        for j, (x, y) in enumerate(test_loader):
            y = y.cpu()
            y_pred = classifier(model(x)).cpu()
            ys.append(y)
            preds.append(torch.argmax(y_pred, dim=-1))

    ys = torch.cat(ys)
    preds = torch.cat(preds)

    logger.info(f'Test acc: {accuracy_score(ys.numpy(), preds.numpy())}')

    print("ys:", ys.numpy())

    confusion = confusion_matrix(ys.numpy(), preds.numpy())
    logger.info('Confusion Matrix:')
    logger.info(confusion)

    # confusion = confusion[1:, 1:]
    # logger.info(f'4 tone accuracy: {np.trace(confusion) / np.sum(confusion)}')


if __name__ == '__main__':
    if args.action == 'train':
        train()
    elif args.action == 'test':
        test()
    else:
        raise RuntimeError(f"Unknown action: {args.action}")
