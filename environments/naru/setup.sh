#!/bin/bash
sudo apt install -y build-essential libpq-dev gcc python3-de

cd ./environments/naru
conda create -y -n naru python=3.9.7
. activate naru

python3 -m pip install -r ./requirements.txt

cd ../../naru/

mkdir ./models
mkdir ./results
