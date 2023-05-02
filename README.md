# Machine Unlearning For Learned Databases

[![License: Apache](https://img.shields.io/badge/License-Apache-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Introduction

This repository contains the code for experimental analysis of data deletion in learned database systems. For this purpose, we study three different learned database systems: [DBEst++](https://github.com/qingzma/DBEst_MDN.git), [Naru](https://github.com/naru-project/naru.git), and [TVAE](https://github.com/sdv-dev/CTGAN.git). 

## Setup

To start installing the packages, run the environmental setup for each application. 

```bash
bash ./environments/dbest/setup.sh
bash ./environments/naru/setup.sh
bash ./environments/tvae/setup.sh
```


## Datasets

We use three real-world datasets for our evaluations: Census, Forest, and DMV.

## Experiments

To increase reproducibility, we have created experimental pipelines for each applications. For census, and forest, you can find related exp_census.py and exp_forest.py scripts. For dmv, there are bash scripts for each dataset to run training/evaluating commands. 
