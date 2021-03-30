from train.utils import onehot_encode

NUM_CLASSES = 6
EMBD_DIM = 128
IN_PLANES = 16
WAV_DIR = '/NASdata/AudioData/AISHELL-ASR-SSB/SPEECHDATA/'
SPEAKER_EMBEDDING_DIR = '/mingback/students/tjy/lanqi/extracted_embeddings/'
CACHE_DIR = 'exp/cache/'

PHONE_TO_ID = {
    'sil': 0,
    'a': 1,
    'ai': 2,
    'an': 3,
    'ang': 4,
    'ao': 5,
    'b': 6,
    'c': 7,
    'ch': 8,
    'd': 9,
    'e': 10,
    'ei': 11,
    'en': 12,
    'eng': 13,
    'er': 14,
    'f': 15,
    'g': 16,
    'h': 17,
    'i': 18,
    'ia': 19,
    'ian': 20,
    'iang': 21,
    'iao': 22,
    'ie': 23,
    'in': 24,
    'ing': 25,
    'iong': 26,
    'iu': 27,
    'j': 28,
    'k': 29,
    'l': 30,
    'm': 31,
    'n': 32,
    'o': 33,
    'ong': 34,
    'ou': 35,
    'p': 36,
    'q': 37,
    'r': 38,
    's': 39,
    'sh': 40,
    't': 41,
    'u': 42,
    'ua': 43,
    'uai': 44,
    'uan': 45,
    'uang': 46,
    'ui': 47,
    'un': 48,
    'uo': 49,
    'v': 50,
    'van': 51,
    've': 52,
    # 'ue': 52,
    "vn": 53,
    'w': 54,
    'x': 55,
    'y': 56,
    'z': 57,
    'zh': 58,
}

N_PHONES = len(list(set(list(PHONE_TO_ID.values()))))

PHONE_TO_ONEHOT = {
    p: onehot_encode(i, N_PHONES).astype('float32') for p, i in PHONE_TO_ID.items()
}
