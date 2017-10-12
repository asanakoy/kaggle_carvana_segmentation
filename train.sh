#!/usr/bin/env bash
# This script will run training for all the networks.
set -e # abort if any command fails
export PYTHONPATH=$(pwd):$PYTHONPATH

pushd asanakoy
bash train_scratch.sh -b1 -g4
bash train_vgg11v1.sh -b1 -g4
popd

# TODO: albu, ternaus
