export KALDI_ROOT=/home/kaldi
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

# ali_path=ssb_align/exp/tri_ali/
ali_path=/mingback/students/tjy/std-mandarin/scripts/exp/tri5a_sp_ali/
stage=1

mkdir -p alignments
if [ $stage -le 1 ]; then
    for f in $ali_path/ali.*.gz; do
        filename="${f##*/}"
        ctmfile=alignments/$filename.ctm
        if [ ! -f $ctmfile ]; then
            echo "$f -> $ctmfile"
            alignfile=$ali_path/$filename.ali
            zcat $f > $alignfile
            ali-to-phones --ctm-output $ali_path/final.mdl ark:$alignfile $ctmfile
        else
            echo "Skipping $f because $ctmfile already exists"
        fi
    done
fi

if [ $stage -le 2 ]; then
    cat alignments/*ctm > phone_ctm.ctm
    python ali_to_phone.py $ali_path/phones.txt phone_ctm.ctm phone_ctm.txt
fi
