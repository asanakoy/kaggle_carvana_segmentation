#!/usr/bin/env bash
# This script will run predict on test for all the networks and assemble a final submission file.
set -e # abort if any command fails
export PYTHONPATH=$(pwd):$PYTHONPATH

pushd asanakoy
bash predict_scratch.sh
bash predict_vgg11v1.sh
popd

# TODO: albu, ternaus

echo "Generate final ensemble"
python generate_sub_final_ensemble.py -j=4
# TODO: fix van in final prediction using albu's script
# or include this fix in generate_sub_final_ensemble.py
