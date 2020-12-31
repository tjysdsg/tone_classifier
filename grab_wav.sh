data_dir=/NASdata/AudioData/mandarin/AISHELL-2/iOS/data/wav
utt=$1
wav=${utt}.wav
spk=${utt:1:5}
wav_file=/NASdata/AudioData/mandarin/AISHELL-2/iOS/data/wav/${spk}/${wav}
echo $wav_file
mkdir -p tmp/
cp $wav_file tmp/