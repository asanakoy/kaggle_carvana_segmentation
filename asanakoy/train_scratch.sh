#!/usr/bin/env bash
set -e # abort if any command fails

echo "TRAIN SCRATCH"
echo "---"

epochs=250
models_dir=$(python -c "import json, os; print os.path.expanduser(json.load(open('../config/config.json', 'r'))['models_dir'])")

usage() { echo "Usage: $0 -b BATCH_SIZE -g GACC -e EPOCHS" 1>&2; exit 1; }

while getopts ":f:b:g:e:n" o; do
    case "${o}" in
        b)
            BATCH=${OPTARG}
            ;;
        g)
            gacc=${OPTARG}
            ;;
        e)
            epochs=${OPTARG}
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


for ((FOLD=0; FOLD<7; FOLD++)); do

    echo "=========="
    echo "FOLD $FOLD"
    echo "BATCH $BATCH"
    echo "gacc $gacc"
    echo "epochs $epochs"
    echo "=========="
    echo ""

    o_dir="${models_dir}/scratch_s1993_im1024_aug1_fold${FOLD}.7"

    python run_train.py -b=$BATCH -gacc=$gacc -f=$FOLD -nf=7 -fv=1 \
        --lr=0.005 -opt=sgd --decay_step=100 --decay_gamma=0.5 \
        -aug=2 --weight_decay=0.0005 \
        -o="${o_dir}" --epochs=$epochs --no_cudnn
done
