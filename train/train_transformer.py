import argparse
import json
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from train.dataset.dataset import EmbeddingDataset
from train.modules.transformers import TransEncoder
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from train.utils import set_seed
import os
import torch
import torch.nn as nn
from train.modules.model_spk import ResNet34StatsPool

EMBD_DIM = 128
IN_PLANES = 16
NUM_CLASSES = 4

# create output dir
SAVE_DIR = 'transformer'
os.makedirs(f'exp/{SAVE_DIR}', exist_ok=True)

parser = argparse.ArgumentParser(description='Deep Speaker Embedding, SGD, ReduceLROnPlateau')
# dataset
parser.add_argument('--data_dir', default='feats', type=str)
parser.add_argument('--data_name', default='train', type=str)
parser.add_argument('-j', '--workers', default=20, type=int)
parser.add_argument('-b', '--batch_size', default=64, type=int)
parser.add_argument('--val_data_name', default='val', type=str)

# others
parser.add_argument('--epochs', default=500, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--seed', default=3007123, type=int)
parser.add_argument('--gpu', default='0,1,2,3,4,5,6,7', type=str)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

set_seed(args.seed)


def load_embedding_model(epoch: int):
    print(f'loading exp/embedding/model_{epoch}.pkl')
    model = ResNet34StatsPool(IN_PLANES, EMBD_DIM).cuda()
    checkpoint = torch.load(f'exp/embedding/model_{epoch}.pkl')
    model.load_state_dict(checkpoint['model'])
    return model


# load and freeze embedding model
embd_model = load_embedding_model(115)
embd_model.eval()

for param in embd_model.parameters():
    param.requires_grad = False

# train dataset
utts = [line.split()[0] for line in open(f'feats/{args.data_name}/wav.scp')]
utt2tones = json.load(open('utt2tones.json'))
train_loader = DataLoader(
    EmbeddingDataset(utts, utt2tones, embd_model), batch_size=args.batch_size, num_workers=args.workers,
    pin_memory=True,
)

# val dataset
val_utts = [line.split()[0] for line in open(f'feats/{args.val_data_name}/wav.scp')]
val_loader = DataLoader(
    EmbeddingDataset(val_utts, utt2tones, embd_model), batch_size=args.batch_size, num_workers=args.workers,
    pin_memory=True,
)

# model, optimizer, criterion, scheduler, trainer
model = TransEncoder(num_classes=NUM_CLASSES, embedding_size=EMBD_DIM)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
criterion = nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, patience=4, verbose=True)
trainer = create_supervised_trainer(model, optimizer, criterion)

# evaluator
val_metrics = {
    "accuracy": Accuracy(),
    "loss": Loss(criterion)
}
evaluator = create_supervised_evaluator(model, metrics=val_metrics)


@trainer.on(Events.ITERATION_COMPLETED(every=100))
def log_training_loss(trainer):
    print(f"Epoch[{trainer.state.epoch}] Loss: {trainer.state.output:.2f}")


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    print(
        f"Training Results - "
        f"Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}"
    )


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    print(
        f"Validation Results - "
        f"Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}"
    )
    scheduler.step(metrics['loss'])


trainer.run(train_loader, max_epochs=500)
