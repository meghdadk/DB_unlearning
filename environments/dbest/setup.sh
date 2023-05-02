#!/bin/bash
sudo apt install -y build-essential libpq-dev gcc python3-de

cd ./environments/dbest
conda create -y -n dbest python=3.9.7
. activate dbest

python3 -m pip install -r ./requirements.txt

cd ../../dbest/

mkdir ./models
mkdir ./benchmark
mkdir ./results
