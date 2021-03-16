import argparse
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

# validation dataset
parser.add_argument('--val_data_name', default='test', type=str)
# learning rate scheduler
parser.add_argument('--lr', default=None, type=float)
parser.add_argument('--lr_patience', default=4, type=int)
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


def create_transformer():
    model = nn.Transformer(d_model=EMBD_DIM)
    return model


def main():
    embd_model = load_embedding_model(115)
    embd_model.eval()

    for param in embd_model.parameters():
        param.requires_grad = False

    model = create_transformer()


if __name__ == '__main__':
    main()
