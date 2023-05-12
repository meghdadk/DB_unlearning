# Machine Unlearning For Learned Databases

[![License: Apache](https://img.shields.io/badge/License-Apache-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Introduction

This repository contains the code for experimental analysis of data deletion in learned database systems. For this purpose, we study three different learned database systems: 

[DBEst++](https://www.cidrdb.org/cidr2021/papers/cidr2021_paper15.pdf): Approximate Query Processing using Mixture Density Networks,

[Naru](https://www.vldb.org/pvldb/vol13/p279-yang.pdf): Cardinality Estimation using Deep Autoregressive Networks

[TVAE](https://proceedings.neurips.cc/paper_files/paper/2019/file/254ed7d2de3b23ab10936522dd547b78-Paper.pdf): Data Generation using Tabular Variational AutoEncoders

## Setup

To start installing the packages, run the environmental setup for each application. 

```bash
bash ./environments/dbest/setup.sh
bash ./environments/naru/setup.sh
bash ./environments/tvae/setup.sh
```


## Datasets

We use three real-world datasets for our evaluations: Census, Forest, and DMV. You can download the versions we use from [here](https://drive.google.com/file/d/1bWbgxpyyITYMWF7LBnWHut7EKIqadxyY/view?usp=share_link)

## Experiments

To increase reproducibility, we have created experimental pipelines for each applications. For census, and forest, you can find related exp_census.py and exp_forest.py scripts. For dmv, there are bash scripts for each dataset to run training/evaluating commands. 


## References

We have used the codes from the below repositories which are the official implementations of the applications we have studied. 

[DBEst++](https://github.com/qingzma/DBEst_MDN.git)

[Naru](https://github.com/naru-project/naru.git)

[TVAE](https://github.com/sdv-dev/CTGAN.git)
