#!/usr/bin/env bash
# This script will run predict on test for all the networks and assemble a final submission file.

bash asanakoy/predict_scratch.sh
bash asanakoy/predict_vgg11v1.sh

# TODO: albu, ternaus

python generate_sub_final_ensemble.py -j=4
# TODO: fix van in final prediction using albu's script
# or include this fix in generate_sub_final_ensemble.py
