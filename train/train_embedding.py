import argparse
from tqdm import trange
import numpy as np
from train.utils import set_seed
from sklearn.metrics import confusion_matrix, accuracy_score
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train.modules.model_spk import ResNet34StatsPool
from train.dataset.dataset import SpectrogramDataset, collate_fn_pad
from utils import AverageMeter, accuracy, save_checkpoint, save_ramdom_state, get_lr, change_lr

# create output dir
SAVE_DIR = 'embedding'
NUM_CLASSES = 5
os.makedirs(f'exp/{SAVE_DIR}', exist_ok=True)

parser = argparse.ArgumentParser(description='Deep Speaker Embedding, SGD, ReduceLROnPlateau')
# dataset
parser.add_argument('--data_dir', default='feats', type=str)
parser.add_argument('--data_name', default='train', type=str)
parser.add_argument('-j', '--workers', default=20, type=int)
parser.add_argument('-b', '--batch_size', default=64, type=int)
parser.add_argument('--val_data_name', default='val', type=str)
parser.add_argument('--test_data_name', default='test', type=str)
# model backbone
parser.add_argument('--in_planes', default=16, type=int)
parser.add_argument('--embd_dim', default=128, type=int)
parser.add_argument('--dropout', default=0.5, type=float)
# optimizer
parser.add_argument('--momentum', default=0.95, type=float)
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float)
# learning rate scheduler
parser.add_argument('--lr', default=None, type=float)
parser.add_argument('--warm_up_epoch', default=3, type=int)
parser.add_argument('--lr_patience', default=4, type=int)
# others
parser.add_argument('action', type=str, default='train', nargs='?')
parser.add_argument('--epochs', default=500, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--seed', default=3007123, type=int)
parser.add_argument('--gpu', default='0,1,2,3,4,5,6,7', type=str)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
set_seed(args.seed)


def create_dataloader(data_dir):
    wavscp = [line.split() for line in open(os.path.join(data_dir, 'wav.scp'))]
    utt2spk = {line.split()[0]: int(line.split()[1]) for line in open(os.path.join(data_dir, 'utt2spk'))}
    return DataLoader(
        SpectrogramDataset(wavscp, utt2spk), batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=True, collate_fn=collate_fn_pad,
    )


# data loaders
train_loader = create_dataloader(f'feats/{args.data_name}')
val_dataloader = create_dataloader(f'feats/{args.val_data_name}')
test_dataloader = create_dataloader(f'feats/{args.test_data_name}')

# models
model = ResNet34StatsPool(
    args.in_planes, args.embd_dim, dropout=args.dropout, total_step=args.epochs
).cuda()
classifier = nn.Linear(args.embd_dim, NUM_CLASSES).cuda()

# criterion, optimizer, scheduler
criterion = nn.CrossEntropyLoss().cuda()
lr = args.lr if args.lr else 0.1 * args.batch_size / 256
optimizer = torch.optim.SGD(
    list(model.parameters()) + list(classifier.parameters()), lr=lr, momentum=args.momentum, weight_decay=args.wd
)
batch_per_epoch = len(train_loader)
lr_lambda = lambda x: lr / (batch_per_epoch * args.warm_up_epoch) * (x + 1)
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
    # torch.cuda.set_rng_state_all(checkpoint['torch_cuda'])
else:
    print(str(model) + '\n' + str(classifier) + '\n')

model = nn.DataParallel(model)


def train():
    lr_change, total_lr_change = 0, 4
    for epoch in range(start_epoch, epochs):
        losses, acc = AverageMeter(), AverageMeter()
        model.train()
        classifier.train()

        # progress bar
        t = trange(len(train_loader))
        t.set_description(f'epoch {epoch}')

        for i, (feats, label) in enumerate(train_loader):
            if epoch < args.warm_up_epoch:
                change_lr(optimizer, lr_lambda(len(train_loader) * epoch + i))

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
        print(
            '\nEpoch %d\t  Loss %.4f\t  Accuracy %3.3f\t  lr %f\t  acc_val %3.3f\n'
            % (epoch, losses.avg, acc.avg, get_lr(optimizer), acc_val)
        )

        last_lr = get_lr(optimizer)
        scheduler.step(losses.avg)
        new_lr = get_lr(optimizer)

        lr_change = lr_change + 1 if last_lr != new_lr else lr_change
        if lr_change == total_lr_change:
            os.system(f'cd exp/{SAVE_DIR} && ln -s model_{epochs}.pkl final.pkl')
            return


def validate() -> float:
    print('============== VALIDATING ==============')
    model.eval()
    classifier.eval()

    ys = []
    preds = []
    with torch.no_grad():
        for j, (x, y) in enumerate(val_dataloader):
            y = y.cpu()
            y_pred = classifier(model(x)).cpu()
            ys.append(y)
            preds.append(torch.argmax(y_pred, dim=-1))

    ys = torch.cat(ys)
    preds = torch.cat(preds)
    print('Confusion Matrix:')
    print(confusion_matrix(ys.numpy(), preds.numpy()))

    return accuracy_score(ys.numpy(), preds.numpy())


def test():
    print('============== TESTING ==============')
    model.eval()
    classifier.eval()

    ys = []
    preds = []
    with torch.no_grad():
        for j, (x, y) in enumerate(test_dataloader):
            y = y.cpu()
            y_pred = classifier(model(x)).cpu()
            ys.append(y)
            preds.append(torch.argmax(y_pred, dim=-1))

    ys = torch.cat(ys)
    preds = torch.cat(preds)

    print(f'Accuracy: {accuracy_score(ys.numpy(), preds.numpy())}')
    print('Confusion Matrix:')
    print(confusion_matrix(ys.numpy(), preds.numpy()))


if __name__ == '__main__':
    if args.action == 'train':
        train()
    elif args.action == 'test':
        test()
    else:
        raise RuntimeError(f"Unknown action: {args.action}")
