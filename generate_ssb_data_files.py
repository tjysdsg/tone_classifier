import os

SSB_DATA_DIR = '/NASdata/AudioData/AISHELL-ASR-SSB/SPEECHDATA/'
OUTPUT_DIR = 'SSB'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    text = []
    with open('aishell-asr-ssb-annotations.txt') as f:
        for line in f:
            tokens = line.replace('\n', '').split()
            utt = tokens[0].split('.')[0]  # remove .wav extension
            trans = ''.join(tokens[1::2])
            text.append('\t'.join([utt, trans]) + '\n')
    with open(f'{OUTPUT_DIR}/text', 'w') as f:
        f.writelines(text)

    utt2spk = []
    wavscp = []
    for speaker in os.scandir(SSB_DATA_DIR):
        if not speaker.is_dir():
            continue
        spk = speaker.name
        print(f'SPEAKER {spk}')
        for wav in os.scandir(speaker.path):
            wavpath = wav.path
            wavname = wav.name
            utt = wavname.split('.')[0]

            wavscp.append('\t'.join([utt, wavpath]) + '\n')
            utt2spk.append('\t'.join([utt, spk]) + '\n')

    with open(f'{OUTPUT_DIR}/wav.scp', 'w') as f:
        f.writelines(wavscp)
    with open(f'{OUTPUT_DIR}/utt2spk', 'w') as f:
        f.writelines(utt2spk)


if __name__ == '__main__':
    main()
