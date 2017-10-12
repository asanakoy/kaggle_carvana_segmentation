#!/usr/bin/env bash
set -e # abort if any command fails

epochs=250
should_gen_sub=1
models_dir=$(python -c "import json, os; print os.path.expanduser(json.load(open('../config/config.json', 'r'))['models_dir'])")

usage() { echo "Usage: $0 -f FOLD -b BATCH_SIZE -g GACC -e EPOCHS -n\n -n: to skip submission generation" 1>&2; exit 1; }

while getopts ":f:b:g:e:n" o; do
    case "${o}" in
        f)
            i=${OPTARG}
            ((s >= 0 || s <= 6)) || usage
            ;;
        b)
            BATCH=${OPTARG}
            ;;
        g)
            gacc=${OPTARG}
            ;;
        e)
            epochs=${OPTARG}
            ;;
        n)
            should_gen_sub=0
            ;;
        *)
            echo "Unknown argument ${o}"
            usage
            ;;
    esac
done
shift $((OPTIND-1))
if [ -z "${i}" ] || [ -z "${BATCH}" ] || [ -z "${gacc}" ] ; then
    echo "Not all reauired arguments were specified"
    usage
fi

((BATCH*gacc == 4)) ||
{ echo "Wrong batch size and GACC! BATCH * GACC must be equal 4! "; exit 1; }


for ((FOLD=i; FOLD<i+1; FOLD++)); do

    echo "=========="
    echo "FOLD $FOLD"
    echo "BATCH $BATCH"
    echo "gacc $gacc"
    echo "epochs $epochs"
    echo "should_gen_sub $should_gen_sub"
    echo "=========="
    echo ""

    o_dir="${models_dir}/scratch_s1993_im1024_aug1_fold${FOLD}.7"

    python run_train.py -b=$BATCH -gacc=$gacc -f=$FOLD -nf=7 -fv=1 \
        --lr=0.005 -opt=sgd --decay_step=100 --decay_gamma=0.5 \
        -aug=2 --weight_decay=0.0005 \
        -o="${o_dir}" --epochs=$epochs --no_cudnn

    ((should_gen_sub)) && python generate_sub.py -b=$BATCH \
        -o "${o_dir}" --no_cudnn
done
