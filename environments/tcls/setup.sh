#!/bin/bash
sudo apt install -y build-essential libpq-dev gcc python3-de

cd ./environments/tcls
conda create -y -n tvae python=3.9.7
. activate tcls

python3 -m pip install -r ./requirements.txt

cd ../../tcls/

mkdir ./checkpoints
mkdir 
