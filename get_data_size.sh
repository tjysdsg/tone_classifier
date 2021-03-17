for i in 0 1 2 3 4; do
  echo "Size of feats/${i}:"
  ls feats/$i | wc -l
done