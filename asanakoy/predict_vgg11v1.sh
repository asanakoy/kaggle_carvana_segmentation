#!/usr/bin/env bash
set -e # abort if any command fails

echo "PREDICT VGG11v1"
echo "---"

models_dir=$(python -c "import json, os; print os.path.expanduser(json.load(open('../config/config.json', 'r'))['models_dir'])")

BATCH=2 # batch size
for ((FOLD=0; FOLD<7; FOLD++)); do
    echo "=========="
    echo "VGG11v1 FOLD $FOLD"
    echo "BATCH $BATCH"
    echo "=========="
    echo ""

    o_dir="${models_dir}/vgg11v1_s1993_im1024_gacc1_aug1_v2fold${FOLD}.7_noreg"
    echo "o_dir=${o_dir}"

    python generate_sub.py -b=$BATCH -o "${o_dir}" -net=vgg11v1 --no_cudnn
done

python generate_sub_average.py --net_name=vgg11v1 -j=4
