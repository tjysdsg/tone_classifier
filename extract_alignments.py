from pydub import AudioSegment
from os.path import join as pjoin
import os
import json

data_root = '/NASdata/AudioData/mandarin/AISHELL-2/iOS/data/wav/'


def time_range_to_file(inpath: str, outpath: str, start: float, duration: float):
    wav = AudioSegment.from_wav(inpath)

    end = start + duration
    start *= 1000  # milliseconds
    end *= 1000  # milliseconds

    seg = wav[start:end]
    seg.export(outpath, format="wav", parameters=["-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000"])


if __name__ == "__main__":
    align = {i: [] for i in range(5)}
    with open('phone_ctm.txt') as f:
        for line in f:
            tokens = line.split()
            phone = tokens[-1]
            tone = phone.split('_')[-1]
            if not tone.isnumeric():
                continue
            tone = int(tone)
            filename = tokens[0]
            start = float(tokens[2])
            dur = float(tokens[3])
            align[tone].append([filename, phone, start, dur])
    json.dump(align, open('align.json', 'w'))
    
    for i in range(5):
        tonedir = pjoin('data', f'{i}')
        os.makedirs(tonedir, exist_ok=True)
        for j, e in enumerate(align[i]):
            filename, phone, start, dur = e
            spk = filename[1:6]
            time_range_to_file(
                pjoin(data_root, spk, f'{filename}.wav'),
                pjoin(tonedir, f"{j}_{filename}_{phone}.wav"),
                start,
                dur,
            )