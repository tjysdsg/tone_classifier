import argparse
from tqdm import trange
import numpy as np
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from embedding.modules.model_spk import ResNet34StatsPool
from embedding.dataset import SpectrogramDataset, collate_fn_pad
from embedding.utils import AverageMeter, accuracy, save_checkpoint, save_ramdom_state, get_lr, change_lr

# create output dir
SAVE_DIR = 'embedding'
os.makedirs(f'exp/{SAVE_DIR}', exist_ok=True)

parser = argparse.ArgumentParser(description='Deep Speaker Embedding, SGD, ReduceLROnPlateau')
# dataset
parser.add_argument('--data_dir', default='feats', type=str)
parser.add_argument('--data_name', default='train', type=str)
parser.add_argument('--dur_range', default=[2, 4], nargs='+', type=int)
parser.add_argument('-j', '--workers', default=20, type=int)
parser.add_argument('-b', '--batch_size', default=64, type=int)
parser.add_argument('--spk_per_batch', default=64, type=int)

# data augmentation
parser.add_argument('--data_aug', default=True, type=bool)
parser.add_argument('--snr_range', default=[0, 20], nargs='+', type=int)
# validation dataset
parser.add_argument('--val_data_name', default='test', type=str)
parser.add_argument('--val_dur_range', default=[8, 8], nargs='+', type=int)
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
parser.add_argument('--lr_patience', default=10, type=int)
# loss type
parser.add_argument('--loss_type', default='CrossEntropy', type=str)
# others
parser.add_argument('--epochs', default=500, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--seed', default=3007123, type=int)
parser.add_argument('--gpu', default='0,1,2,3,4,5,6,7', type=str)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

SEED = args.seed
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# training dataset
utt2wav = [line.split() for line in open(f'feats/{args.data_name}/wav.scp')]
spk2utt = {int(line.split()[0]): line.split()[1:] for line in open(f'{args.data_dir}/{args.data_name}/spk2utt')}
utt2spk = {line.split()[0]: int(line.split()[1]) for line in open(f'{args.data_dir}/{args.data_name}/utt2spk')}
NUM_CLASSES = 4

dataset = SpectrogramDataset(utt2wav, utt2spk, num_classes=NUM_CLASSES)
# batch_sampler = WavBatchSampler(dataset, args.dur_range, shuffle=True, batch_size=args.batch_size, drop_last=True)
train_loader = DataLoader(
    dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
    collate_fn=collate_fn_pad
)

# validation dataset
val_wavscp = [line.split() for line in open(f'feats/{args.val_data_name}/wav.scp')]
val_utt2spk = {line.split()[0]: line.split()[1] for line in open(f'feats/{args.val_data_name}/utt2spk')}
val_dataset = SpectrogramDataset(val_wavscp, val_utt2spk, num_classes=NUM_CLASSES)
# batch_sampler = WavBatchSampler(
#     val_dataset, args.val_dur_range, shuffle=False, batch_size=args.batch_size, drop_last=False
# )
val_dataloader = DataLoader(
    val_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
    collate_fn=collate_fn_pad
)

# models
model = ResNet34StatsPool(
    args.in_planes, args.embd_dim, dropout=args.dropout, total_step=args.epochs
).cuda()
classifier = nn.Sequential(nn.Linear(args.embd_dim, NUM_CLASSES), nn.Softmax(dim=-1)).cuda()

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
classifier.train()


def main():
    lr_change, total_lr_change = 0, 4
    for epoch in range(start_epoch, epochs):
        losses, top1 = AverageMeter(), AverageMeter()
        model.train()

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

            prec1 = accuracy(outputs.data, label)
            losses.update(loss.data.item(), feats.size(0))
            top1.update(prec1[0].data.item(), feats.size(0))

            # update progress bar
            t.set_postfix(loss=losses.val, loss_avg=losses.avg, acc=top1.val, acc_avg=top1.avg,
                          lr=get_lr(optimizer))
            t.update()

        save_checkpoint(f'exp/{SAVE_DIR}', epoch, model, classifier, optimizer, scheduler)
        # noinspection PyUnresolvedReferences
        save_ramdom_state(
            f'exp/{SAVE_DIR}', random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all()
        )

        acc = validate()
        print(
            '\nEpoch %d\t  Loss %.4f\t  Accuracy %3.3f\t  lr %f\t  acc_val %3.3f\n'
            % (epoch, losses.avg, top1.avg, get_lr(optimizer), acc)
        )

        last_lr = get_lr(optimizer)
        scheduler.step(losses.avg)
        new_lr = get_lr(optimizer)

        lr_change = lr_change + 1 if last_lr != new_lr else lr_change
        if lr_change == total_lr_change:
            os.system(f'cd exp/{SAVE_DIR} && ln -s model_{epochs}.pkl final.pkl')
            return


def validate() -> float:
    print('=' * 25)
    model.eval()

    acc = AverageMeter()
    with torch.no_grad():
        for j, (x, y) in enumerate(val_dataloader):
            y = y.cpu()
            y_pred = classifier(model(x)).cpu()
            acc.update(accuracy(y_pred, y)[0].data.item(), y.size(0))

    return acc.avg


if __name__ == '__main__':
    main()
