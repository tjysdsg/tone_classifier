import os
from train.config import CACHE_DIR

cache_dir = os.path.join(CACHE_DIR, 'spectro')
scp_path = os.path.join(cache_dir, 'wav.scp')

os.system(f'cp {scp_path} {scp_path}.backup')

scp = open(scp_path, 'w')

for spk in os.scandir(cache_dir):
    if spk.is_dir():
        for f in os.scandir(spk.path):
            utt = f.name.split('.')[0]
            path = f.path
            scp.write(f'{utt}\t{path}\n')

scp.close()
