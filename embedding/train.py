import argparse
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader

import embedding.modules.back_classifier as classifiers
import embedding.modules.model_spk as models
from embedding.dataset import WavDataset, WavBatchSampler, logFbankCal
from embedding.utils.spk_veri_metric import SVevaluation
from embedding.utils.utils import AverageMeter, accuracy, save_checkpoint, save_ramdom_state, get_lr, change_lr

# import
# from dataset.dataset_ffsvc import WavDataset
# auto stop when resume

parser = argparse.ArgumentParser(description='Deep Speaker Embedding, SGD, ReduceLROnPlateau')
parser.add_argument('--save_dir', type=str)
# training dataset
parser.add_argument('--data_name', default='train_vox1', type=str)
parser.add_argument('--dur_range', default=[2, 4], nargs='+', type=int)
parser.add_argument('-j', '--workers', default=20, type=int)
parser.add_argument('-b', '--batch_size', default=256, type=int)
parser.add_argument('--batch_sampler', default='WavBatchSampler', type=str)
parser.add_argument('--spk_per_batch', default=64, type=int)

# data augmentation
parser.add_argument('--data_aug', default=True, type=bool)
parser.add_argument('--snr_range', default=[0, 20], nargs='+', type=int)
# validation dataset
parser.add_argument('--val_data_name', default='test_vox', type=str)
parser.add_argument('--val_dur_range', default=[8, 8], nargs='+', type=int)
# eer and cost
parser.add_argument('--ptar', default=[0.01, 0.001], nargs='+', type=float)
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
# model classifier
parser.add_argument('--classifier', default='Linear', type=str)
parser.add_argument('--angular_m', default=0.1, type=float)
parser.add_argument('--angular_s', default=32, type=float)
# optimizer
parser.add_argument('--momentum', default=0.95, type=float)
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float)
# learning rate scheduler
parser.add_argument('--lr', default=None, type=float)
parser.add_argument('--warm_up_epoch', default=5, type=int)
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
torchaudio.initialize_sox()

# feature
featCal = logFbankCal(sample_rate=args.fs,
                      n_fft=args.fft,
                      win_length=int(args.win_len * args.fs),
                      hop_length=int(args.hop_len * args.fs),
                      n_mels=args.mels).cuda()
featCal.eval()

# training dataset
utt2wav = [line.split() for line in open('data/%s/wav.scp' % args.data_name)]
spk2int = {line.split()[0]: i for i, line in enumerate(open('data/%s/spk2utt' % args.data_name))}
spk2utt = {line.split()[0]: line.split()[1:] for line in open('data/%s/spk2utt' % args.data_name)}
utt2spk = {line.split()[0]: spk2int[line.split()[1]] for line in open('data/%s/utt2spk' % args.data_name)}
noise_list = {'noise': [i.strip('\n') for i in open('data/envir/noise_wav_list')],
              'music': [i.strip('\n') for i in open('data/envir/music_wav_list')],
              'babb': [i.strip('\n') for i in open('data/envir/speech_wav_list')],
              'reverb': [i.strip('\n') for i in open('data/envir/simu_rir_list')]}

dataset = WavDataset(utt2wav, utt2spk, args.fs, is_aug=args.data_aug, snr=args.snr_range, noise_list=noise_list,
                     channel=1)
batch_sampler = WavBatchSampler(dataset, args.dur_range, shuffle=True, batch_size=args.batch_size, drop_last=True)
train_loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=args.workers, pin_memory=True)

# validation dataset
val_dataset = WavDataset([line.split() for line in open('data/%s/wav.scp' % args.val_data_name)], fs=args.fs)
batch_sampler = WavBatchSampler(val_dataset, args.val_dur_range, shuffle=False, batch_size=args.batch_size,
                                drop_last=False)
val_dataloader = DataLoader(val_dataset, batch_sampler=batch_sampler, num_workers=args.workers, pin_memory=True)

# EER & cost calculator
utt = [line.split()[0] for line in open('data/%s/wav.scp' % args.val_data_name)]
eer_cal = SVevaluation(utt, utt, 'data/%s/trials' % args.val_data_name, ptar=args.ptar)

# models
model = getattr(models, args.model)(args.in_planes, args.embd_dim, dropout=args.dropout,
                                    total_step=args.epochs).cuda()  # resnet34
classifier = getattr(classifiers, args.classifier)(args.embd_dim, len(spk2int), m=args.angular_m, s=args.angular_s,
                                                   device_id=[i for i in range(len(args.gpu.split(',')))]).cuda()

criterion = nn.CrossEntropyLoss().cuda()
# optimizer
lr = args.lr if args.lr else 0.1 * args.batch_size / 256
optimizer = torch.optim.SGD(list(model.parameters()) + list(classifier.parameters()),
                            lr=lr, momentum=args.momentum, weight_decay=args.wd)

# learning rate scheduler
batch_per_epoch = len(train_loader)
lr_lambda = lambda x: lr / (batch_per_epoch * args.warm_up_epoch) * (x + 1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.lr_patience, factor=0.1)

# other parameters
save_dir = args.save_dir
os.system('mkdir -p exp/%s' % save_dir)

epochs, start_epoch = args.epochs, args.start_epoch
if start_epoch != 0:
    print('Load exp/%s/model_%d.pkl' % (save_dir, start_epoch - 1))
    checkpoint = torch.load('exp/%s/model_%d.pkl' % (save_dir, start_epoch - 1))
    model.load_state_dict(checkpoint['model'])
    classifier.load_state_dict(checkpoint['classifier'])
    if not args.lr:
        optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    checkpoint = torch.load('exp/%s/random_state.pkl' % (save_dir))
    random.setstate(checkpoint['random'])
    np.random.set_state(checkpoint['np'])
    torch.set_rng_state(checkpoint['torch'])
    # torch.cuda.set_rng_state_all(checkpoint['torch_cuda'])
    logs = open('exp/%s/train.out' % save_dir, 'a')
else:
    logs = open('exp/%s/train.out' % save_dir, 'w')
    logs.write(str(model) + '\n' + str(classifier) + '\n')

model = nn.DataParallel(model)
classifier.train()


def main():
    lr_change, total_lr_change = 0, 4

    for epoch in range(start_epoch, epochs):
        losses, top1 = AverageMeter(), AverageMeter()
        model.train()
        end = time.time()

        for i, (feats, is_spec_aug, key) in enumerate(train_loader):
            #             if i<532:
            #                 continue
            data_time = time.time() - end

            if epoch < args.warm_up_epoch:
                change_lr(optimizer, lr_lambda(len(train_loader) * epoch + i))

            feats, key = feats.cuda(), key.cuda()

            outputs = classifier(model(featCal(feats, is_spec_aug).transpose(1, 2)), key)
            loss = criterion(outputs, key)

            # outputs = classifier(model(featCal(feats, is_spec_aug)), key)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prec1 = accuracy(outputs.data, key)
            losses.update(loss.data.item(), feats.size(0))
            top1.update(prec1[0].data.item(), feats.size(0))

            batch_time = time.time() - end
            end = time.time()

            logs.write('Length [%d]\t' % (feats.size()[1] / args.hop_len / args.fs) +
                       'Epoch [%d][%d/%d]\t ' % (epoch, i + 1, len(train_loader)) +
                       'Time [%.3f/%.3f]\t' % (batch_time, data_time) +
                       'Loss %.4f %.4f\t' % (losses.val, losses.avg) +
                       'Accuracy %3.3f %3.3f\t' % (top1.val, top1.avg) +
                       'LR %.6f\n' % get_lr(optimizer))
            logs.flush()

        save_checkpoint('exp/%s' % save_dir, epoch, model, classifier, optimizer, scheduler)
        save_ramdom_state('exp/%s' % save_dir, random.getstate(),
                          np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())

        eer, cost = validate()
        logs.write('Epoch %d\t  Loss %.4f\t  Accuracy %3.3f\t  lr %f\t  EER %.4f\t  cost %.4f\n'
                   % (epoch, losses.avg, top1.avg, get_lr(optimizer), eer, cost))

        last_lr = get_lr(optimizer)
        scheduler.step(losses.avg)
        new_lr = get_lr(optimizer)

        lr_change = lr_change + 1 if last_lr != new_lr else lr_change
        if lr_change == total_lr_change:
            os.system('cd exp/%s && ln -s model_%d.pkl final.pkl' % (save_dir, epoch))
            return


def validate():
    model.eval()
    embd = np.zeros([0, args.embd_dim])
    with torch.no_grad():
        for j, (feat, utt) in enumerate(val_dataloader):
            embd = np.concatenate((embd, model(featCal(feat.cuda()).transpose(1, 2)).cpu().numpy()))
    eer, cost = eer_cal.eer_cost(embd, embd)
    return eer, cost


if __name__ == '__main__':
    main()
