#!/usr/bin/env bash
set -e # abort if any command fails

echo "PREDICT SCRATCH"
echo "---"

models_dir=$(python -c "import json, os; print os.path.expanduser(json.load(open('../config/config.json', 'r'))['models_dir'])")

BATCH=2 # batch size
for ((FOLD=0; FOLD<7; FOLD++)); do
    echo "=========="
    echo "Unet from scratch"
    echo "FOLD $FOLD"
    echo "BATCH $BATCH"
    echo "=========="
    echo ""

    o_dir="${models_dir}/scratch_s1993_im1024_aug1_fold${FOLD}.7"
    echo "o_dir=${o_dir}"

    python generate_sub.py -b=$BATCH -o "${o_dir}" --no_cudnn
done

python generate_sub_average.py --net_name=scratch -j=4
