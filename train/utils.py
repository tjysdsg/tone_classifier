import os
import torch
import numpy as np
import random
from train.modules.model_spk import ResNet34StatsPool


def load_embedding_model(epoch: int, in_planes: int, embd_dim: int) -> ResNet34StatsPool:
    print(f'loading exp/embedding/model_{epoch}.pkl')
    model = ResNet34StatsPool(in_planes, embd_dim).cuda()
    checkpoint = torch.load(f'exp/embedding/model_{epoch}.pkl')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_embedding_model(epoch: int, in_planes: int, embd_dim: int):
    print(f'loading exp/embedding/model_{epoch}.pkl')
    model = ResNet34StatsPool(in_planes, embd_dim)
    checkpoint = torch.load(f'exp/embedding/model_{epoch}.pkl')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    for param in model.parameters():
        param.requires_grad = False
    return model


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_ramdom_state(chk_dir, ramdom_state, np_stats, torch_state, torch_cuda_state):
    torch.save({'random': ramdom_state,
                'np': np_stats,
                'torch': torch_state,
                'torch_cuda': torch_cuda_state
                }, os.path.join(chk_dir, 'random_state.pkl'))


def save_checkpoint(chk_dir, epoch, model, classifier, optimizer, scheduler=None, lr=None):
    torch.save({'model': model.module.state_dict(),
                'classifier': classifier.state_dict() if classifier else None,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'lr': lr
                }, os.path.join(chk_dir, 'model_%d.pkl' % epoch))


def save_transformer_checkpoint(chk_dir, epoch, model, optimizer, scheduler):
    torch.save(
        {
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, os.path.join(chk_dir, 'model_%d.pkl' % epoch)
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target) -> float:
    total = 0
    correct = 0
    _, predicted = torch.max(output.data, 1)
    total += target.size(0)
    correct += (predicted == target).sum().item()
    return correct / total


def masked_accuracy(output, target, padding_mask) -> float:
    total = 0
    correct = 0
    batch_size = output.size(0)
    for i in range(batch_size):
        o = output[i]
        t = target[i]
        mask = padding_mask[i]

        _, pred = torch.max(o.data, 1)
        pred[mask] = 0
        t[mask] = -1

        total += t.size(0) - mask.sum().item()
        correct += (pred == t).sum().item()

    return correct / total


def change_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def separate_resnet_bn_paras(modules):
    all_parameters = modules.parameters()
    paras_only_bn = []

    for pname, p in modules.named_parameters():
        if pname.find('bn') >= 0:
            paras_only_bn.append(p)

    paras_only_bn_id = list(map(id, paras_only_bn))
    paras_wo_bn = list(filter(lambda p: id(p) not in paras_only_bn_id, all_parameters))

    return paras_only_bn, paras_wo_bn


if __name__ == '__main__':
    y_pred = torch.as_tensor(
        [
            [[0.3, 0.7], [0.7, 0.3], [0.3, 0.7]],
            [[0.3, 0.7], [0.3, 0.7], [0.7, 0.3]],
            [[0.3, 0.7], [0.3, 0.7], [0.3, 0.7]]
        ], dtype=torch.float
    )
    y = torch.as_tensor(
        [
            [1, 0, 1],
            [1, 0, 0],
            [1, 0, 0],
        ],
        dtype=torch.long
    )
    mask = torch.as_tensor(
        [
            [True, True, True],
            [True, True, False],
            [True, False, False],
        ],
        dtype=torch.bool
    )

    assert masked_accuracy(y_pred, y, ~mask) == 5 / 6
