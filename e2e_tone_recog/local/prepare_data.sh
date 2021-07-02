#!/usr/bin/env bash

# Create text and wav.scp from AISHELL-3 with text containing tone transcripts

. ./path.sh || exit 1;

train_utts=../data/train_utts.json
test_utts=../data/test_utts.json
output_dir=data
utt2tones=../utt2tones.json

. tools/parse_options.sh || exit 1;

if [ ! -f ${utt2tones} ]; then
  echo "${utt2tones} does not exist" || exit 1
fi

python ../generate_e2e_data.py --utts=${train_utts} --utt2tones=${utt2tones} --output-dir=${output_dir}/train || exit 1
python ../generate_e2e_data.py --utts=${test_utts} --utt2tones=${utt2tones} --output-dir=${output_dir}/test || exit 1

echo "local/prepare_data.sh succeeded"
