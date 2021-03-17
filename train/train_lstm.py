import argparse
from sklearn.model_selection import train_test_split
import json
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from train.dataset.dataset import EmbeddingDataset, collate_fn_pack_pad
from train.modules.lstm import LSTMNet
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from train.utils import set_seed, load_embedding_model
import os
import torch
import torch.nn as nn

EMBD_DIM = 128
IN_PLANES = 16
NUM_CLASSES = 4

# create output dir
SAVE_DIR = 'transformer'
os.makedirs(f'exp/{SAVE_DIR}', exist_ok=True)

parser = argparse.ArgumentParser('train transformer', description='Train embedding on transformer')
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

# load and freeze embedding model
embd_model = load_embedding_model(115, IN_PLANES, EMBD_DIM)
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
model = LSTMNet(num_classes=NUM_CLASSES, embedding_size=EMBD_DIM)
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
