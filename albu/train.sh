#!/usr/bin/env bash
set -e # abort if any command fails
source activate py35_albu

pushd src
PYTHONPATH=$(pwd):$PYTHONPATH python train.py
popd

source deactivate
