#! /usr/bin/env python3
import os, argparse, numpy as np
import torch, torch.nn as nn, modules.model_spk as models
from torch.utils.data import DataLoader
from dataset import SpectrogramDataset, logFbankCal

parser = argparse.ArgumentParser(description='Speaker Embedding Extraction')
parser.add_argument('--save_dir', type=str)
# validation dataset
parser.add_argument('--val_data_name', default='test_vox', type=str)
parser.add_argument('--scp_name', default='wav', type=str)
# acoustic feature
parser.add_argument('--fs', default=16000, type=int)
parser.add_argument('--fft', default=512, type=int)
parser.add_argument('--mels', default=64, type=int)
parser.add_argument('--win_len', default=0.025, type=float)
parser.add_argument('--hop_len', default=0.01, type=float)
# model backbone
parser.add_argument('--model', default='ResNet34StatsPool', type=str)
parser.add_argument('--in_planes', default=16, type=int)
parser.add_argument('--embd_dim', default=128, type=int)
parser.add_argument('--dropout', default=0.5, type=float)
# others
parser.add_argument('--gpu', default='1', type=str)
parser.add_argument('--verbose', default=True, type=bool)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# feature
featCal = logFbankCal(sample_rate=args.fs,
                      n_fft=args.fft,
                      win_length=int(args.win_len * args.fs),
                      hop_length=int(args.hop_len * args.fs),
                      n_mels=args.mels).cuda()
featCal.eval()

# dataset
val_dataset = SpectrogramDataset([line.split() for line in open('data/%s/%s.scp' % (args.val_data_name, args.scp_name))],
                                 fs=args.fs)
val_dataloader = DataLoader(val_dataset, num_workers=4, shuffle=False, batch_size=1)

# models
model = getattr(models, args.model)(args.in_planes, args.embd_dim, dropout=args.dropout).cuda()
model.load_state_dict(torch.load('exp/%s/final.pkl' % args.save_dir)['model'])
model = nn.DataParallel(model)
model.eval()

embd = np.zeros([len(val_dataloader), args.embd_dim])
with torch.no_grad():
    for j, (feat, utt) in enumerate(val_dataloader):
        embd[j, :] = model(featCal(feat.cuda())).cpu().numpy()
        if args.verbose:
            print(j, utt, feat.shape[1])

np.save('exp/%s/%s_%s' % (args.save_dir, args.val_data_name, args.scp_name), embd)
