#!/usr/bin/env bash
# This script will run all trainings, run all predicts and assemble a final submission file.

bash train.sh || exit 1
bash predict.sh || exit 1
