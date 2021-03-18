import os
import argparse
from tqdm import trange
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import json
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from train.dataset.dataset import EmbeddingDataset, collate_fn_pack_pad
from train.modules.transformers import TransEncoder
from train.utils import set_seed, AverageMeter, accuracy, save_checkpoint, save_ramdom_state, get_lr
import torch
import torch.nn as nn
from train.modules.model_spk import ResNet34StatsPool

EMBD_DIM = 128
IN_PLANES = 16
NUM_CLASSES = 4

# create output dir
SAVE_DIR = 'transformer'
os.makedirs(f'exp/{SAVE_DIR}', exist_ok=True)

parser = argparse.ArgumentParser(description='Train embedding on transformer')
parser.add_argument('--data_dir', default='feats', type=str)
parser.add_argument('--data_name', default='train', type=str)
parser.add_argument('-j', '--workers', default=20, type=int)
parser.add_argument('-b', '--batch_size', default=64, type=int)
parser.add_argument('--val_data_name', default='val', type=str)
parser.add_argument('--epochs', default=500, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--seed', default=3007123, type=int)
args = parser.parse_args()

set_seed(args.seed)


def load_embedding_model(epoch: int):
    print(f'loading exp/embedding/model_{epoch}.pkl')
    model = ResNet34StatsPool(IN_PLANES, EMBD_DIM)
    checkpoint = torch.load(f'exp/embedding/model_{epoch}.pkl')
    model.load_state_dict(checkpoint['model'])
    return model


# load and freeze embedding model
embd_model = load_embedding_model(121)
embd_model.eval()

for param in embd_model.parameters():
    param.requires_grad = False

utt2tones = json.load(open('utt2tones.json'))
utts = list(utt2tones.keys())
utts_train, utts_test = train_test_split(utts, test_size=0.25)
utts_train, utts_val = train_test_split(utts_train, test_size=0.1)

# train dataset
train_loader = DataLoader(
    EmbeddingDataset(utts_train, utt2tones, embd_model), batch_size=args.batch_size, num_workers=args.workers,
    pin_memory=True, collate_fn=collate_fn_pack_pad,
)

# val dataset
val_loader = DataLoader(
    EmbeddingDataset(utts_val, utt2tones, embd_model), batch_size=args.batch_size, num_workers=args.workers,
    pin_memory=True, collate_fn=collate_fn_pack_pad,
)

# model, optimizer, criterion, scheduler, trainer
model = TransEncoder(num_classes=NUM_CLASSES, embedding_size=EMBD_DIM).cuda()
model = nn.DataParallel(model)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
criterion = nn.CrossEntropyLoss().cuda()
scheduler = ReduceLROnPlateau(optimizer, patience=4, verbose=True)


def main():
    for epoch in range(500):
        losses, top1 = AverageMeter(), AverageMeter()
        model.train()

        # progress bar
        t = trange(len(train_loader))
        t.set_description(f'epoch {epoch}')

        for i, (feats, label) in enumerate(train_loader):
            feats, label = feats.cuda(), label.cuda()

            outputs = model(feats)
            loss = criterion(outputs, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prec1 = accuracy(outputs.data, label)
            losses.update(loss.data.item(), feats.size(0))
            top1.update(prec1[0].data.item(), feats.size(0))

            # update progress bar
            t.set_postfix(loss=losses.val, loss_avg=losses.avg, acc=top1.val, acc_avg=top1.avg,
                          lr=get_lr(optimizer))
            t.update()

        # TODO: save_checkpoint(f'exp/{SAVE_DIR}', epoch, model, classifier, optimizer, scheduler)
        # TODO: save_ramdom_state(
        #     f'exp/{SAVE_DIR}', random.getstate(), np.random.get_state(), torch.get_rng_state(),
        #     torch.cuda.get_rng_state_all()
        #  )

        acc = validate()
        print(
            '\nEpoch %d\t  Loss %.4f\t  Accuracy %3.3f\t  lr %f\t  acc_val %3.3f\n'
            % (epoch, losses.avg, top1.avg, get_lr(optimizer), acc)
        )

        scheduler.step(losses.avg)


def validate() -> float:
    print('=' * 25)
    model.eval()

    acc = AverageMeter()
    ys = []
    preds = []
    with torch.no_grad():
        for j, (x, y) in enumerate(val_loader):
            y = y.cpu()
            y_pred = model(x).cpu()
            ys.append(y)
            preds.append(torch.argmax(y_pred, dim=-1))

            acc.update(accuracy(y_pred, y)[0].data.item(), y.size(0))
    ys = torch.cat(ys)
    preds = torch.cat(preds)
    print('Confusion Matrix:')
    print(confusion_matrix(ys.numpy(), preds.numpy()))

    return acc.avg


if __name__ == '__main__':
    main()
