#!/usr/bin/env bash
# This script will run predict on test for all the networks and assemble a final submission file.
set -e # abort if any command fails
source activate py27
export PYTHONPATH=$(pwd):$PYTHONPATH

pushd asanakoy
bash predict_scratch.sh
bash predict_vgg11v1.sh
popd

pushd albu
bash predict.sh
popd

pushd ternaus
bash predict.sh
popd

echo "Generate final ensemble"
python generate_sub_final_ensemble.py -j=4
source deactivate
