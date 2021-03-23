import json

AISHELL_TRANS_TXT = '/NASdata/AudioData/mandarin/AISHELL-2/iOS/data/new.txt'


def main():
    ssb2utt = {}
    with open('ssbutt.txt') as f:
        for line in f:
            utt, ssb = line.replace('\n', '').split('|')
            ssb2utt[ssb] = utt

    filtered_wavs = set()
    with open('wav_filtered.scp') as f:
        for line in f:
            utt = line.split('\t')[0]
            filtered_wavs.add(utt)

    aishell_trans = {}
    with open(AISHELL_TRANS_TXT) as f:
        for line in f:
            tokens = line.replace('\n', '').split()
            if len(tokens) != 2:
                print(f'{tokens} invalid')
                continue
            utt, trans = tokens
            aishell_trans[utt] = trans

    ssb2aishell = {}
    with open('aishell-asr-ssb-annotations.txt') as f:
        for line in f:
            tokens = line.replace('\n', '').split()
            ssb = tokens[0].split('.')[0]  # remove .wav extension
            utt = ssb2utt.get(ssb)
            if utt is None or utt not in filtered_wavs:
                continue
            trans = ''.join(tokens[1::2])

            if trans == aishell_trans[utt]:
                ssb2aishell[ssb] = utt
            else:
                print(f'{ssb}/{utt} has unmatched transcript: {trans} vs. {aishell_trans[utt]}')

    json.dump(ssb2aishell, open('ssb2aishell.json', 'w'))


if __name__ == '__main__':
    main()
