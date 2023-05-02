#!/bin/bash
sudo apt install -y build-essential libpq-dev gcc python3-de

cd ./environments/tvae
conda create -y -n tvae python=3.9.7
. activate tvae

python3 -m pip install -r ./requirements.txt

cd ../../tvae/

mkdir ./models
mkdir ./results
