#!/usr/bin/env bash
set -e # abort if any command fails
source activate py35_albu
PYTHONPATH=$(pwd):$PYTHONPATH python train.py
source deactivate
