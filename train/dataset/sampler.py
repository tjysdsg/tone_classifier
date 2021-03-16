import copy
import random
from collections import defaultdict
from torch.utils.data import RandomSampler, SequentialSampler


class WavBatchSampler(object):
    def __init__(self, dataset, tlen_range, shuffle=False, batch_size=1, drop_last=False):
        self.tlen_range = tlen_range
        self.batch_size = batch_size
        self.drop_last = drop_last

        if shuffle:
            self.sampler = RandomSampler(dataset)
        else:
            self.sampler = SequentialSampler(dataset)

    def _renew(self):
        return [], random.uniform(self.tlen_range[0], self.tlen_range[1])

    def __iter__(self):
        batch, tlen = self._renew()
        for idx in self.sampler:
            batch.append((idx, tlen))
            if len(batch) == self.batch_size:
                yield batch
                batch, tlen = self._renew()
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class WavBalancedBatchSampler(object):
    def __init__(self, dataset, tlen_range, spk_per_batch=32, batch_size=1, drop_last=False):

        self.tlen_range = tlen_range
        self.spk_per_batch = spk_per_batch

        self.drop_last = drop_last
        self.batch_size = batch_size

        self.spk2idx = defaultdict(list)
        for i, (utt, _) in enumerate(dataset.wav_scp):
            spk_id = dataset.utt2label[utt]
            self.spk2idx[spk_id].append(i)

        self.sampler = SequentialSampler(dataset)
        self.all_idx = None

    def _arrange_idx(self):
        self.all_idx = []
        left_spk2idx = copy.deepcopy(self.spk2idx)
        for spk in left_spk2idx:
            random.shuffle(left_spk2idx[spk])
        utt_per_spk = self.batch_size // self.spk_per_batch

        while left_spk2idx:
            if len(left_spk2idx.keys()) >= self.spk_per_batch:
                for spk in random.sample(left_spk2idx.keys(), self.spk_per_batch):
                    self.all_idx += self._select_utt(left_spk2idx, spk, utt_per_spk)
            else:
                for spk in list(left_spk2idx.keys()):
                    self.all_idx += self._select_utt(left_spk2idx, spk, utt_per_spk)

    def _select_utt(self, left_spk2idx, spk, n):
        if len(left_spk2idx[spk]) <= n:
            return left_spk2idx.pop(spk)
        else:
            return [left_spk2idx[spk].pop() for i in range(n)]

    def _renew(self):
        return [], random.uniform(self.tlen_range[0], self.tlen_range[1])

    def __iter__(self):
        self._arrange_idx()

        batch, tlen = self._renew()
        for i in self.sampler:
            batch.append((self.all_idx[i], tlen))
            if len(batch) == self.batch_size:
                yield batch
                batch, tlen = self._renew()
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
