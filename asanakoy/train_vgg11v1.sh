#!/usr/bin/env bash
set -e # abort if any command fails

BATCH=2
gacc=$((4/BATCH))
models_dir=$(python -c "import json, os; print os.path.expanduser(json.load(open('../config/config.json', 'r'))['models_dir'])")

((BATCH == 1 || BATCH == 2 || BATCH == 4)) ||
{ echo "Wrong batch size! Only size 1, 2 or 4 is possible"; exit 1; }

# Calculate train car areas. To split on folds according to area
python calc_car_areas.py -j=2

for ((FOLD=0; FOLD<7; FOLD++)); do
    echo "=========="
    echo "VGG11v1 FOLD $FOLD"
    echo "BATCH $BATCH"
    echo "GACC $gacc"
    echo "=========="
    echo ""

    o_dir="${models_dir}/vgg11v1_s1993_im1024_gacc1_aug1_v2fold${FOLD}.7_noreg"

    python run_train.py -net=vgg11v1 -imsize=1024 -b=$BATCH -gacc=$gacc -f=$FOLD -nf=7 -fv=2 \
        --lr=0.0001 -opt=adam --cyclic_lr=20 -aug=1  --no_cudnn --weight_decay=0 \
    -o "$o_dir" --epochs=15

    python run_train.py -net=vgg11v1 -imsize=1024 -b=$BATCH -gacc=$gacc -f=$FOLD -nf=7 -fv=2 \
        --lr=0.0001 -opt=adam --cyclic_lr=20 -aug=0  --no_cudnn --weight_decay=0 \
        -o "$o_dir" --epochs=60
done
