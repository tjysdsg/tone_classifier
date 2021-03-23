export LC_ALL=C

pushd /NASdata/pc_backup/jiayan/close_talk_manderin/

root_dir=/mingback/students/tjy/tone_classifier/ssb_align/
data_dir=/mingback/students/tjy/tone_classifier/SSB/
mfccdir=$root_dir/mfcc
mfcc_conf=/mingback/students/tjy/tone_classifier/conf/mfcc.conf

mkdir -p $root_dir
mkdir -p $mfccdir

. ./cmd.sh
. ./path.sh

stage=2

if [ $stage -le 1 ]; then
  utils/fix_data_dir.sh $data_dir || exit 1;
fi

if [ $stage -le 2 ]; then
  steps/make_mfcc.sh --mfcc-config $mfcc_conf --cmd "$train_cmd" --nj 40 $data_dir $root_dir/exp/make_mfcc/ $mfccdir || exit 1;
  steps/compute_cmvn_stats.sh $data_dir $root_dir/exp/make_mfcc/ $mfccdir || exit 1;
fi

if [ $stage -le 3 ]; then
  steps/align_fmllr.sh --cmd "$train_cmd" --nj 100 $data_dir data/lang exp/tri14 $root_dir/exp/tri_ali || exit 1;
fi

popd