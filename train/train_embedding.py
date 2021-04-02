import argparse
import sys
import numpy as np
from typing import Tuple
import json
from tqdm import trange
from train.utils import (
    set_seed, create_logger, AverageMeter, accuracy, save_checkpoint, get_lr,
)
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train.modules.models import *
from train.dataset.dataset import create_dataloader
from train.config import NUM_CLASSES, EMBD_DIM, IN_PLANES, MAX_GRAD_NORM

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='Training embedding')
parser.add_argument('--save_dir', type=str)

parser.add_argument('-j', '--workers', default=20, type=int)
parser.add_argument('-b', '--batch_size', default=32, type=int)

parser.add_argument('--data_dir', default='data', type=str)
parser.add_argument('--val_data_name', default='val', type=str)
parser.add_argument('--test_data_name', default='test', type=str)

parser.add_argument('--use_attention', default=False, action='store_true')

parser.add_argument('--include_segment_feats', default=False, action='store_true')
parser.add_argument('--include_spk', default=False, action='store_true')
parser.add_argument('--context_size', default=0, type=int)

parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--warm_up_epoch', default=3, type=int)
parser.add_argument('--lr_patience', default=1, type=int)

parser.add_argument('--tone_pattern', default='', type=str)  # for testing on specific tone patterns
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

INCLUDE_SEGMENT_FEATS = args.include_segment_feats
INCLUDE_SPK = args.include_spk
CONTEXT_SIZE = args.context_size
print(
    f'Using segment features: {INCLUDE_SEGMENT_FEATS}\n'
    f'Using speaker embedding: {INCLUDE_SPK}\n'
    f'Context size: {CONTEXT_SIZE}'
)

logger = create_logger('train_embedding', f'exp/{SAVE_DIR}/{args.action}_{args.start_epoch}.log')
logger.info(" ".join(sys.argv))  # save entire command for reproduction

utt2tones: dict = json.load(open('utt2tones.json'))

# data loaders
data_train: list = json.load(open(f'{DATA_DIR}/train_utts.json'))
data_test: list = json.load(open(f'{DATA_DIR}/test_utts.json'))

# tone pattern
tone_pattern = args.tone_pattern.split(',')
if len(tone_pattern) < 2:
    print('Not matching tone pattern')
    tone_pattern = None
else:
    tone_pattern = [int(tp) for tp in tone_pattern]
    print(f'Matching tone pattern: {tone_pattern}')


def _create_loader(utt_list):
    return create_dataloader(
        utt_list, utt2tones, include_segment_feats=INCLUDE_SEGMENT_FEATS, context_size=CONTEXT_SIZE,
        include_spk=INCLUDE_SPK, batch_size=args.batch_size, n_workers=args.workers,
        tone_pattern=tone_pattern,
    )


train_loader = _create_loader(data_train)
test_loader = _create_loader(data_test)
print('train size:', len(train_loader) * args.batch_size)
print('test size:', len(test_loader) * args.batch_size)

# models
if args.use_attention:
    inner_model = ResNet34AttStatsPool(IN_PLANES, EMBD_DIM, dropout=0.5).cuda()
else:
    inner_model = ResNet34StatsPool(IN_PLANES, EMBD_DIM, dropout=0.5).cuda()

model = EmbeddingModel(
    inner_model, EMBD_DIM, NUM_CLASSES, include_segment_feats=INCLUDE_SEGMENT_FEATS, context_size=CONTEXT_SIZE,
    include_spk=INCLUDE_SPK,
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


def step(batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = batch[0]
    y = batch[1]

    durs = None
    onehots = None
    spk_embd = None
    if len(batch) >= 3:
        durs = batch[2]

    if len(batch) >= 4:
        onehots = batch[3]

    if len(batch) >= 5:
        spk_embd = batch[4]

    y = y.cuda()
    y_pred = model(x, durs, onehots, spk_embd)
    return x, y, y_pred


def train():
    for epoch in range(start_epoch, epochs):
        losses, acc = AverageMeter(), AverageMeter()
        model.train()

        # progress bar
        t = trange(len(train_loader))
        t.set_description(f'epoch {epoch}')

        for i, batch in enumerate(train_loader):
            x, y, y_pred = step(batch)
            loss = criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()

            if args.use_attention:
                _ = nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                _ = nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            optimizer.step()

            losses.update(loss.data.item(), y.size(0))
            acc.update(accuracy(y_pred.data, y), y.size(0))

            # update progress bar
            t.set_postfix(loss=losses.val, loss_avg=losses.avg, acc=acc.val, acc_avg=acc.avg, lr=get_lr(optimizer))
            t.update()

        save_checkpoint(f'exp/{SAVE_DIR}', epoch, model, optimizer, scheduler)

        acc_val = validate(test_loader)
        logger.info(
            '\nEpoch %d\t  Loss %.4f\t  Accuracy %3.3f\t  lr %f\t  acc_val %3.3f\n'
            % (epoch, losses.avg, acc.avg, get_lr(optimizer), acc_val)
        )

        scheduler.step(losses.avg)


def infer(dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()

    ys = []
    preds = []
    with torch.no_grad():
        for j, batch in enumerate(dataloader):
            _, y, y_pred = step(batch)
            y = y.cpu()
            y_pred = y_pred.cpu()
            ys.append(y)
            preds.append(torch.argmax(y_pred, dim=-1))

    ys = torch.cat(ys).numpy()
    preds = torch.cat(preds).numpy()
    return ys, preds


def validate(dataloader: DataLoader) -> float:
    ys, preds = infer(dataloader)

    acc = accuracy_score(ys, preds)
    logger.info(f'Accuracy: {acc}')

    confusion = confusion_matrix(ys, preds)
    logger.info(f'\n{confusion}')

    report = classification_report(ys, preds, digits=4)
    logger.info(f'\n{report}')
    return acc


def infer_embeddings(dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # register hook to get embeddings from model
    embeddings = []

    def hook(_, __, output):
        embeddings.append(output.cpu().numpy())

    model.model1.register_forward_hook(hook)
    labels, y_pred = infer(dataloader)
    embeddings = np.vstack(embeddings)
    return labels, y_pred, embeddings


def tsne(dataloader: DataLoader):
    from sklearn.manifold import TSNE
    from matplotlib import pyplot as plt
    import seaborn as sns

    labels, _, embeddings = infer_embeddings(dataloader)
    embeddings = np.vstack(embeddings)
    tsne = TSNE(n_components=2, perplexity=50, n_jobs=10).fit_transform(embeddings)
    x = tsne[:, 0]
    y = tsne[:, 1]
    sns.scatterplot(x=x, y=y, hue=labels)
    plt.savefig(f'exp/{SAVE_DIR}/tsne.png')


def pca(dataloader: DataLoader):
    from sklearn.decomposition import PCA
    from matplotlib import pyplot as plt
    import seaborn as sns

    labels, _, embeddings = infer_embeddings(dataloader)
    embeddings = np.vstack(embeddings)
    tsne = PCA(n_components=2).fit_transform(embeddings)
    x = tsne[:, 0]
    y = tsne[:, 1]
    sns.scatterplot(x=x, y=y, hue=labels)
    plt.savefig(f'exp/{SAVE_DIR}/pca.png')


if __name__ == '__main__':
    if args.action == 'train':
        model = nn.DataParallel(model)
        train()
    elif args.action == 'test':
        model = nn.DataParallel(model)
        validate(test_loader)
    elif args.action == 'tsne':
        tsne(test_loader)
    elif args.action == 'pca':
        pca(test_loader)
    else:
        raise RuntimeError(f"Unknown action: {args.action}")
