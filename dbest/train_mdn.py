import argparse
import os
import sys
import math
import scipy
import random
import time
import dill
import numpy as np
import pandas as pd
import pandasql as ps
import wandb

import matplotlib.pyplot as plt
from concurrent import futures
from copy import deepcopy
from collections import Counter
from itertools import cycle
from numpy.random import choice
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from scipy import integrate,stats
from math import e

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.multiprocessing import Pool
import category_encoders as ce


from mdn import MDN, DenMDN
import Dataset
from sqlParser import Parser
import utils
from utils import mdn_loss, mse_kd_loss, adjust_learning_rate


seed = 1234
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


dill._dill._reverse_typemap['ClassType'] = type
#os.environ["WANDB_MODE"] = "offline"

"""To work with multiprocessing, uncomment this part \
   This, however, slows down training. 
try:
    torch.multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass
"""


def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--x_att', type=str, help='x-attribute', required=True)
    parser.add_argument('--y_att', type=str, help='y-attribute', required=True)
    parser.add_argument('--dataset', type=str, required=True,
			choices=['census','forest','dmv'], help='Dataset.')
    parser.add_argument('--datafile', type=str, help='path to the csv datafile', required=True)
    parser.add_argument('--filters', type=str, help='path to the json filters')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train','finetune', 'NegGrad', 'NegGrad+', 'SCRUB', 'linear-sgda', 
                                 'retrain', 'bcu', 'rbcu', 'bcu-distill', 'stale'])
    parser.add_argument('--tr_frac', type=float, default=1, help='the size of transfer-set between 0-1')
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--opt', type=str, default='adam', choices=['sgd', 'adam', 'rmsp'], help='optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--del_batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--lr_decay_epochs', type=str, default='10,20,30', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')

    parser.add_argument('--pre_model', type=str, default=None, help='fine-tune on a pre-trained model.')
    parser.add_argument('--remove_weights', action='store_true', help='re-initialize the weights of the model?')
    parser.add_argument('--evaluate', action='store_true', help='whether to evaluate the model')
    parser.add_argument('--eval_likelihood', action='store_true', help='whether to evaluate the likelihood of the model')
    parser.add_argument('--eval_per_epoch', type=int, help='the number i means evaluate every ith epoch. 0 means disabled')
    parser.add_argument('--aggs', type=str, default='sum,count,avg', help='the agg functions to evaluate')
    parser.add_argument('--num_eval_queries', type=int, help='the number of queries to evaluate the model')
    parser.add_argument('--use_pre_queries', action='store_true', help='use the previously generated queries or regenerate')
    parser.add_argument('--eval_deleted', action='store_true', help='evalaute error on the deleted rows')
    parser.add_argument('--compare_hist', action='store_true', help='compare histograms')
    parser.add_argument('--x_val_for_hist', type=str, help='x value to compare histograms')
    parser.add_argument('--bins', type=int, default=None, help='number of bins to plot histograms')

    parser.add_argument('-a', '--alpha', type=float, default=0.5, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=1, help='weight balance for negative losses')
    parser.add_argument('-ms', '--msteps', type=int, default=50, help='number of steps in ascent direction')

    parser.add_argument('--num_hid_layers', type=int, default=2, help='number of hidden layers')
    parser.add_argument('--hid_layer_sizes', type=str, default='128,128', help='comma joined hidden layers sizes')
    parser.add_argument('--num_gaussians', type=int, default=30, help='number of Gaussians in the last layer')


    #Wandb arguments
    parser.add_argument('--wandb_mode', type=str, default='disabled', choices=['online', 'offline', 'disabled'], 
                        help='wandb running mode')
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='the project on wandb to add the runs')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='your wandb user name')
    parser.add_argument('--wandb_run_id', type=str, default=None,
                        help='To resume a previous run with an id')
    parser.add_argument('--wandb_group_name', type=str, default=None,
                        help='Given name to group runs together')

    args = parser.parse_args()
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    layers = args.hid_layer_sizes.split(',')
    args.hid_layer_sizes = list([])
    for l in layers:
        args.hid_layer_sizes.append(int(l))

    aggs = args.aggs.split(',')
    args.aggs = list([])
    for a in aggs:
        args.aggs.append(a)

    assert len(args.hid_layer_sizes) == args.num_hid_layers

    return args

def init_logger(args):

    if args.wandb_group_name is None:
        args.wandb_group_name = args.dataset
    if args.wandb_run_id is not None:
        logger = wandb.init(project=args.wandb_project, entity=args.wandb_entity, id=args.wandb_run_id, mode=args.wandb_mode, resume="must")
    else:
        logger = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                   mode=args.wandb_mode, group=args.wandb_group_name, config=args)

    return logger



def train(args, x_att, y_att, logger=None):
    """Creating a Mixture Density Network

    Args:
        dataset (string): dataset name
        data_path (string): Address to the data file in csv format.
        x_att (list of strings): A list of the names of the x attributes
        y_att (list of strings): A list of the names of the y_attributes

    Note:
        currently, only 1 x attribute and 1 y attribute is supported

    """
    print ("\n")
    print ("-"*60)
    print ("mode: {}\ndata: {}\nprevious-model: {}\n".format(args.mode, args.datafile, args.pre_model))
    print ("===>")
    assert os.path.exists(args.datafile)

    model_name = 'models/' + args.dataset + '_' + \
                    args.mode + '_' + ','.join(x_att) + \
                    '_' + ','.join(y_att) + '_' + \
                    "seed{}".format(seed) + ".dill"

    if args.pre_model is not None:
        assert os.path.exists(args.pre_model)
        model = None 
        with open(args.pre_model, 'rb') as d: 
            model = dill.load(d)

        data_handler = model.dataset
        data_handler.data_path = args.datafile

    if args.pre_model is None:
        #print ("Start preparing data ...")
        t1 = time.time()
        data_handler = Dataset.Data(args.datafile, x_attributes=x_att, y_attributes=y_att, sep=',')    

    x_values, y_values = data_handler.read_data(_return="dictionaries")

    data_handler.create_encoders(data=x_values)
    data_handler.create_frequency_tables(data=x_values)
    data_handler.create_normalizers(data=y_values, min_max=None)


    y_normalized = data_handler.normalize(data=y_values)
    x_encoded = {}
    for key in x_values.keys():
        x_encoded[key] = data_handler.encoders[key].transform(np.asarray(x_values[key]).reshape(-1,1)).toarray()


    #print ("Data is prepared: \n\t1-Encoders are created, \n\t2-data has been normalized, \n\t3-frequency tables are created ")

    if args.pre_model is not None:
        MDN = model
    else:
        MDN = DenMDN(dataset=data_handler, args=args)
    
    t1 = time.time()
    MDN.fit(x_points=x_encoded[data_handler.x_attributes[0]], y_points=y_normalized[data_handler.y_attributes[0]], args=args, logger=logger)
    t2 = time.time()
    print ("Training finished in {} seconds".format(t2-t1))

    with open(model_name,'wb') as dum:
        dill.dump(MDN,dum)

    return MDN, model_name

def delete(args, logger):
    """Delete a part of data from the model/table


    Args:
        model (string): path to a pre-trained DenMDN model

    """
    print ("\n")
    print ("-"*60)
    print ("mode: {}\ndata: {}\nfilters: {}\nprevious-model: {} \n".format(args.mode, args.datafile, args.filters, args.pre_model))
    print ("===>")

    assert os.path.exists(args.pre_model)
    assert os.path.exists(args.filters)

    model = None 
    with open(args.pre_model, 'rb') as d: 
        model = dill.load(d)
    
    if args.datafile is not None:
        assert os.path.exists(args.datafile)
        model.dataset.data_path = args.datafile

    filters_path = args.filters

    data, removed_rows = model.dataset.delete_data(filters_path)

    if args.mode == "retrain":
        transfer_set = data.sample(frac=1)
    else:
        transfer_set = data.sample(frac=args.tr_frac)
    x_values, y_values = model.dataset.get_x_y_dict(transfer_set)
    y_normalized = model.dataset.normalize(data=y_values)
    x_encoded = {}
    for key in x_values.keys():
        x_encoded[key] = model.dataset.encoders[key].transform(np.asarray(x_values[key]).reshape(-1,1)).toarray() 
    

    x_values_del, y_values_del = model.dataset.get_x_y_dict(removed_rows)
    y_normalized_del = model.dataset.normalize(data=y_values_del)
    x_encoded_del = {}
    for key in x_values_del.keys():
        x_encoded_del[key] = model.dataset.encoders[key].transform(np.asarray(x_values_del[key]).reshape(-1,1)).toarray()


    t1 = time.time()
    if args.mode == "stale":
        args.epochs = 0
        model.finetune(x_points=x_encoded[model.dataset.x_attributes[0]],
                y_points=y_normalized[model.dataset.y_attributes[0]],
                x_deletion=x_encoded_del[model.dataset.x_attributes[0]],
                y_deletion=y_normalized_del[model.dataset.y_attributes[0]],
                args=args, keep_weights=True, logger=logger)
    elif args.mode == "retrain":
        model.finetune(x_points=x_encoded[model.dataset.x_attributes[0]],
                y_points=y_normalized[model.dataset.y_attributes[0]],
                x_deletion=x_encoded_del[model.dataset.x_attributes[0]],
                y_deletion=y_normalized_del[model.dataset.y_attributes[0]],
                args=args, keep_weights=False, logger=logger)
    elif args.mode == "finetune":       
        model.finetune(x_points=x_encoded[model.dataset.x_attributes[0]],
                y_points=y_normalized[model.dataset.y_attributes[0]],
                x_deletion=x_encoded_del[model.dataset.x_attributes[0]],
                y_deletion=y_normalized_del[model.dataset.y_attributes[0]],
                args=args, keep_weights=True, logger=logger)
    elif args.mode == "sgda-linear":       
        model.delete_2(x_points=x_encoded[model.dataset.x_attributes[0]],
                y_points=y_normalized[model.dataset.y_attributes[0]],
                x_deletion=x_encoded_del[model.dataset.x_attributes[0]],
                y_deletion=y_normalized_del[model.dataset.y_attributes[0]],
                args=args, keep_weights=True, logger=logger)
    elif args.mode == "SCRUB":       
        model.scrub(x_points=x_encoded[model.dataset.x_attributes[0]],
                y_points=y_normalized[model.dataset.y_attributes[0]],
                x_deletion=x_encoded_del[model.dataset.x_attributes[0]],
                y_deletion=y_normalized_del[model.dataset.y_attributes[0]],
                args=args, keep_weights=True, logger=logger)
    elif args.mode == "NegGrad" or args.mode == "NegGrad+":       
        model.negativegrad(x_points=x_encoded[model.dataset.x_attributes[0]],
                y_points=y_normalized[model.dataset.y_attributes[0]],
                x_deletion=x_encoded_del[model.dataset.x_attributes[0]],
                y_deletion=y_normalized_del[model.dataset.y_attributes[0]],
                args=args, keep_weights=True, logger=logger)
    else:
        raise NotImplementedError()

    t2 = time.time()

    print ("model unlearned in {} seconds".format(t2 - t1))


    model_name = 'models/' + args.dataset + '_' + \
                    args.mode + '_' + ','.join([args.x_att]) + \
                    '_' + ','.join([args.y_att]) + '_' + \
                    "seed{}".format(seed) + ".dill"

    with open(model_name,'wb') as dum:
        dill.dump(model,dum)    

    return model, model_name

if __name__=="__main__":

    args = parser()
    logger = init_logger(args)

    if args.mode == "train":
        model, out_path = train(args, x_att=[args.x_att], y_att=[args.y_att], logger=logger)
        #distill(teacher_model='models/DenMDN_census.dill',
        #    x_att=["native_country"], y_att=["age"])
    else:
        model, out_path = delete(args, logger)


    #if using wandb, save the latest model as well as the filters file
    if isinstance(logger, type(wandb.run)) and logger is not None:
        #artifact = wandb.Artifact('model', type='model')
        #artifact.add_file(out_path)
        #logger.log_artifact(artifact)
        logger.save(out_path)

        if args.filters is not None:
            if os.path.exists(args.filters):
                #artifact = wandb.Artifact('filters', type='file')
                #artifact.add_file(args.filters)
                #logger.log_artifact(artifact)
                logger.save(args.filters)


