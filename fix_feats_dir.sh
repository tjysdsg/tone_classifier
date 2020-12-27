for i in 0 1 2 3 4; do
    echo "Processing feats/${i}"
    find feats/${i}/ -maxdepth 1 -name "*.jpg" -type f > feats/${i}.list
done