import os
import argparse
from collections import defaultdict

args = argparse.ArgumentParser(description='')
args.add_argument('-i', '--utt2spk', default=None, type=str)
args = args.parse_args()

spk2utt = defaultdict(list)
for line in open(args.utt2spk):
    utt, spk = line.split()
    spk2utt[spk].append(utt)

for spk in sorted(spk2utt.keys()):
    print(spk, ' '.join(spk2utt[spk]))
