import json

# 声母
INITIALS = ['b', 'c', 'ch', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 'sh', 't', 'w', 'x', 'y',
            'z', 'zh']
OUTDIR = 'feats'


def main():
    # SSB utterance id to aishell2 utterance id
    ssb2utt = {}
    with open('ssbutt.txt') as f:
        for line in f:
            utt, ssb = line.replace('\n', '').split('|')
            ssb2utt[ssb] = utt

    # utt of filtered wavs
    filtered_wavs = set()
    with open('wav_filtered.scp') as f:
        for line in f:
            utt = line.split('\t')[0]
            filtered_wavs.add(utt)

    # utt to its phone-level transcript
    utt2trans = {}
    with open('aishell-asr-ssb-annotations.txt') as f:
        for line in f:
            tokens = line.replace('\n', '').split()
            utt = tokens[0].split('.')[0]
            utt = ssb2utt.get(utt)
            if utt is None or utt not in filtered_wavs:
                continue
            phones = tokens[2::2]
            # phones = [p.replace('5', '0') for p in phones]  # 轻声 5 -> 0
            # separate initials and finals
            utt2trans[utt] = []
            for p in phones:
                if p[:2] in INITIALS:
                    utt2trans[utt].append(p[:2])
                    final = p[2:]
                elif p[:1] in INITIALS:
                    utt2trans[utt].append(p[:1])
                    final = p[1:]
                else:
                    final = p

                # 去掉儿化
                if 'er' not in final and final[-2] == 'r':
                    utt2trans[utt].append(final.replace('r', ''))
                    utt2trans[utt].append('er5')
                else:
                    utt2trans[utt].append(final)

    # utt to timestamps
    utt2time = {}  # {utt: [start, dur, tone]}
    with open('phone_ctm.txt') as f:
        prev_utt = 'fuck'
        prev_dur = 0
        for line in f:
            tokens = line.split()
            utt = tokens[0]
            if utt not in filtered_wavs:
                continue
            if utt not in utt2time:
                utt2time[utt] = []

            start = float(tokens[2])
            dur = float(tokens[3])
            phone = tokens[4]
            tone = phone.split('_')[-1]

            if tone.isnumeric():
                tone = int(tone)
            if phone in ['$0', 'sil', 'spn']:  # ignore empty phones
                continue

            # triphone
            # NOTE: don't change the value of dur, it's the true duration of this phone
            if utt != prev_utt:
                utt2time[utt].append([start, dur, tone])
            else:
                # include this phone in the previous triphone
                utt2time[utt][-1][1] += dur / 2

                # include previous phone in this triphone
                utt2time[utt].append(
                    [start - prev_dur / 2, prev_dur / 2 + dur, tone]
                )
            prev_dur = dur
            prev_utt = utt

    all_data = []  # list of (tone, utt, phone, start, dur)
    utt2tones = {}  # {utt: [tone, phone, start, dur]}
    for utt, data in utt2time.items():
        trans = utt2trans.get(utt)
        if trans is None:
            continue
        if len(trans) != len(data):
            print(f'WARNING: utt {utt} different length of transcript and timestamps')
            continue

        # collect phone boundaries from ASR and tones from annotations
        for i, p in enumerate(trans):
            tone = p[-1]
            if not tone.isnumeric():  # initials don't have tones, using 0 to represent
                tone = 0
            else:
                tone = int(tone)
                if tone == 5:  # not including light tone
                    continue
            start = data[i][0]
            dur = data[i][1]

            if utt not in utt2tones:
                utt2tones[utt] = []
            utt2tones[utt].append([tone, p, start, dur])
            all_data.append([tone, utt, p, start, dur])

    with open('all_data.json', 'w') as f:
        json.dump(all_data, f)
    with open('utt2tones.json', 'w') as f:
        json.dump(utt2tones, f)


if __name__ == "__main__":
    main()
