from train.utils import onehot_encode

NUM_CLASSES = 6
EMBD_DIM = 128
IN_PLANES = 16
WAV_DIR = '/NASdata/AudioData/AISHELL-ASR-SSB/SPEECHDATA/'
CACHE_DIR = 'exp/cache/'

PHONE_TO_ID = {
    "a": 0,
    "ai": 1,
    "an": 2,
    "ang": 3,
    "ao": 4,
    "b": 5,
    "c": 6,
    "ch": 7,
    "d": 8,
    "e": 9,
    "ei": 10,
    "en": 11,
    "eng": 12,
    "er": 13,
    "f": 14,
    "g": 15,
    "h": 16,
    "i": 17,
    "ia": 18,
    "ian": 19,
    "iang": 20,
    "iao": 21,
    "ie": 22,
    "in": 23,
    "ing": 24,
    "iong": 25,
    "iu": 26,
    "j": 27,
    "k": 28,
    "l": 29,
    "m": 30,
    "n": 31,
    "o": 32,
    "ong": 33,
    "ou": 34,
    "p": 35,
    "q": 36,
    "r": 37,
    "s": 38,
    "sh": 39,
    "t": 40,
    "u": 41,
    "ua": 42,
    "uai": 43,
    "uan": 44,
    "uang": 45,
    "ui": 46,
    "un": 47,
    "uo": 48,
    "v": 49,
    "van": 50,
    "ve": 51,
    "ue": 51,  # <---
    # "vn"
    "w": 52,
    "x": 53,
    "y": 54,
    "z": 55,
    "zh": 56,
}

N_PHONES = len(list(set(list(PHONE_TO_ID.values()))))

PHONE_TO_ONEHOT = {
    p: onehot_encode(i, N_PHONES).astype('float32') for p, i in PHONE_TO_ID.items()
}
