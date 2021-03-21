import os
import argparse
from tqdm import trange
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from train.dataset.dataset import SequentialEmbeddingDataset, collate_sequential_embedding
from train.modules.transformers import TransEncoder
from train.utils import (
    set_seed, AverageMeter, masked_accuracy, save_transformer_checkpoint, get_lr, create_logger,
    load_transformer_data,
)
import torch
import torch.nn as nn
from train.config import NUM_CLASSES, EMBD_DIM, IN_PLANES

# create output dir
SAVE_DIR = 'transformer'
MAX_GRAD_NORM = 10
os.makedirs(f'exp/{SAVE_DIR}', exist_ok=True)

parser = argparse.ArgumentParser(description='Train embedding on transformer')
parser.add_argument('action', type=str, default='train', nargs='?')
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

logger = create_logger('train_transformer', f'exp/{SAVE_DIR}/{args.action}_{args.start_epoch}.log')

utt2tones, utts_train, utts_test, utts_val = load_transformer_data()

# datasets
train_loader = DataLoader(
    SequentialEmbeddingDataset(utts_train, utt2tones, embedding_dir=args.embedding_dir),
    batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, collate_fn=collate_sequential_embedding,
)
val_loader = DataLoader(
    SequentialEmbeddingDataset(utts_val, utt2tones, embedding_dir=args.embedding_dir),
    batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, collate_fn=collate_sequential_embedding,
)
test_loader = DataLoader(
    SequentialEmbeddingDataset(utts_test, utt2tones, embedding_dir=args.embedding_dir),
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


def train():
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

        acc_val = validate()
        logger.info(
            '\nEpoch %d\t  Loss %.4f\t  Accuracy %3.3f\t  lr %f\t  acc_val %3.3f\n'
            % (epoch, float(losses.avg), float(acc.avg), get_lr(optimizer), acc_val)
        )

        scheduler.step(losses.avg)


def validate() -> float:
    logger.info('============== VALIDATING ==============')
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


def test():
    from sklearn.metrics import confusion_matrix

    logger.info('============== TESTING ==============')
    model.eval()

    ys = []
    preds = []
    with torch.no_grad():
        for j, (x, y, lengths) in enumerate(test_loader):
            y_pred, _ = model(x, lengths)
            y = y.cpu()
            y_pred = torch.argmax(y_pred, dim=-1).cpu()

            for i in range(x.shape[0]):
                ys.append(y[i][:lengths[i]])
                preds.append(y_pred[i][:lengths[i]])

    ys = torch.cat(ys)
    preds = torch.cat(preds)

    logger.info('Confusion Matrix:')
    logger.info(confusion_matrix(ys.numpy(), preds.numpy()))


if __name__ == '__main__':
    if args.action == 'train':
        train()
    elif args.action == 'test':
        test()
    else:
        raise RuntimeError(f"Unknown action: {args.action}")
