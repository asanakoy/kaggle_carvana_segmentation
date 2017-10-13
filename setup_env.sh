#!/usr/bin/env bash
conda create -n py35_albu python=3.5
source activate py35_albu

pip install -r ./albu/requirements.txt
pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl
pip install torchvision

source deactivate

#TODO ternaus

conda create -n py27 python=2.7
source activate py27

pip install -r requirements.txt

source deactivate
