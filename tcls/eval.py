import os
import time
import numpy as np
import scipy.stats as st
from itertools import cycle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import models
import datasets
from data import DataProcessor
import utils
from KD import DistillKL


checkpoints = os.listdir('checkpoints/')

DATASET = 'census'
MODEL = 'resnet'
DATA_PATH = '../tabular_data/census/census.csv'
DELETE_FILTER = 'census_filters_selective.json'
COLS = [
    'age','workclass','fnlwgt','education',
    'marital_status','occupation','relationship',
    'race','sex','capital_gain','capital_loss',
    'hours_per_week','native_country'
]
CAT_COLS = ['workclass', 'education', 'occupation', 'relationship', 'race', 'sex', 'native_country']
NUM_COLS = ['age', 'fnlwgt', 'capital_gain', 'capital_loss', 'hours_per_week']
LABEL = 'marital_status'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 1

# Loading data and creating the loaders
data_handler = DataProcessor(DATA_PATH, COLS, label=LABEL, cat_cols=CAT_COLS, num_cols=NUM_COLS, filters_path=DELETE_FILTER)
data = data_handler._read_data()
all_data, reduced_data, deleted_data = data_handler._delete_data(data)

train_loader, val_loader, test_loader = data_handler._prepare_data(data, update_params=True, bs=2048, test_frac=0.1, val_frac=0.1)
forget_loader, _, _ = data_handler._prepare_data(deleted_data, update_params=False, bs=2048, test_frac=0, val_frac=0)
retain_loader, _, _ = data_handler._prepare_data(reduced_data, update_params=False, bs=2048, test_frac=0, val_frac=0)

# Creating the model
if MODEL == 'resnet':
    model = models.ResNet1D(num_blocks=[8, 8, 16], 
                            num_classes=len(data_handler.classes), 
                            data_dim=data_handler.train_size[1]).to(DEVICE)
elif MODEL == 'mlp':
    model = models.MLP(input_size=data_handler.train_size[1], 
                        hidden_size=128, 
                        num_classes=len(data_handler.classes), 
                        dropout_prob=0).to(DEVICE)
else:
    raise ValueError(f'{MODEL} models are not supported yet!')



methods = ['original', 'retrain', 'finetune', 'neggrad', 'neggradplus', 'scrub']
methods = ['neggrad', 'neggradplus']

for method in methods:
    test_accs = []
    retain_accs = []
    forget_accs = []
    for m in checkpoints:
        tokens = m.split('-')
        if (tokens[0] == DATASET \
            and (tokens[3] == method) \
            and ('selective' in m)):
            model.load_state_dict(torch.load('checkpoints/'+m))
            model.eval()

            test_acc, retain_acc, forget_acc = utils.all_tests(model, retain_loader, forget_loader, test_loader, DEVICE)

            test_accs.append(test_acc*100)
            retain_accs.append(retain_acc*100)
            forget_accs.append(forget_acc*100)
            #print (m, test_acc, retain_acc, forget_acc)



    interval_test = st.t.interval(confidence=.95, df=len(test_accs), loc=np.mean(test_accs), scale=st.sem(test_accs))
    interval_retain = st.t.interval(confidence=.95, df=len(retain_accs), loc=np.mean(retain_accs), scale=st.sem(retain_accs))
    interval_forget = st.t.interval(confidence=.95, df=len(forget_accs), loc=np.mean(forget_accs), scale=st.sem(forget_accs))
    string = f"{method} & {np.mean(test_accs):.2f}±{(interval_test[1] - interval_test[0]):.2f} &"
    string += f"{np.mean(retain_accs):.2f}±{(interval_retain[1] - interval_retain[0]):.2f} &"
    string += f"{np.mean(forget_accs):.2f}±{(interval_forget[1] - interval_forget[0]):.2f} \\\\"
    print (string)


