NUM_CLASSES = 4
EMBD_DIM = 128
IN_PLANES = 16
WAV_DIR = '/NASdata/AudioData/AISHELL-ASR-SSB/SPEECHDATA/'
CACHE_DIR = 'exp/cache/'
PRETRAINED_EMBEDDINGS_DIR = 'embeddings'

PHONES = [
    "a",
    "ai",
    "an",
    "ang",
    "ao",
    "b",
    "c",
    "ch",
    "d",
    "e",
    "ei",
    "en",
    "eng",
    "er",
    "f",
    "g",
    "h",
    "i",
    "ia",
    "ian",
    "iang",
    "iao",
    "ie",
    "in",
    "ing",
    "iong",
    "iu",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "ong",
    "ou",
    "p",
    "q",
    "r",
    "s",
    "sh",
    "t",
    "u",
    "ua",
    "uai",
    "uan",
    "uang",
    "ui",
    "un",
    "uo",
    "v",
    "van",
    "ve",
    "vn",
    "w",
    "x",
    "y",
    "z",
    "zh",
]
N_PHONES = len(PHONES)

from train.utils import onehot_encode

PHONE_TO_ONEHOT = {
    p: onehot_encode(i, N_PHONES).astype('float32') for i, p in enumerate(PHONES)
}
