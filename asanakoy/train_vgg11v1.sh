#!/usr/bin/env bash
set -e # abort if any command fails

echo "TRAIN VGG11v1"
echo "---"

models_dir=$(python -c "import json, os; print os.path.expanduser(json.load(open('../config/config.json', 'r'))['models_dir'])")

usage() { echo "Usage: $0 -b BATCH_SIZE -g GACC"; exit 1; }

while getopts ":f:b:g:e:n" o; do
    case "${o}" in
        b)
            BATCH=${OPTARG}
            ;;
        g)
            gacc=${OPTARG}
            ;;
        *)
            echo "Unknown argument ${o}"
            usage
            ;;
    esac
done
shift $((OPTIND-1))
if [ -z "${BATCH}" ] || [ -z "${gacc}" ] ; then
    echo "Not all reauired arguments were specified"
    usage
fi

((BATCH == 1 || BATCH == 2 || BATCH == 4)) ||
{ echo "Wrong batch size! Only size 1, 2 or 4 is possible"; exit 1; }
((BATCH*gacc == 4)) ||
{ echo "Wrong batch size and GACC! BATCH * GACC must be equal 4! "; exit 1; }


# Calculate train car areas. To split on folds according to area
python calc_car_area.py -j=2

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

    echo "Train after 15th epoch"
    echo "-"
    python run_train.py -net=vgg11v1 -imsize=1024 -b=$BATCH -gacc=$gacc -f=$FOLD -nf=7 -fv=2 \
        --lr=0.0001 -opt=adam --cyclic_lr=20 -aug=0  --no_cudnn --weight_decay=0 \
        -o "$o_dir" --epochs=60
done
