#!/usr/bin/env bash

which conda
if [ "$?" -eq 1 ]; then
    wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
    chmod +x Miniconda2-latest-Linux-x86_64.sh
    ./Miniconda2-latest-Linux-x86_64.sh
else
    echo "Miniconda already installed"
fi

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
pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl
pip install torchvision 

source deactivate
