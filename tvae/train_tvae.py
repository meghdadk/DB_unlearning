import argparse
import time
import pickle
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import wandb
import json



from ctgan import TVAESynthesizer
from ctgan import load_demo
from ctgan import data as data_handler
import benchmarking


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=[
            'Train', 'Retrain', 'Finetune', 'Stale', 'NegGrad', 'NegGrad+'],
            help='the evaluation method?')
    parser.add_argument('--dataset', type=str, required=True, choices=[
            'census', 'forest', 'dmv'],
            help='Dataset.')
    parser.add_argument('--datafile', type=str, help='path to the csv datafile', required=True)
    parser.add_argument('--deletedfile', type=str, help='path to the csv deleted file')
    parser.add_argument('--metadatafile', type=str, help='path to the meta data json', required=True)
    parser.add_argument('--model', type=str, help='path to the model to evaluate')
    parser.add_argument('--eval', action='store_true', help='eval deleted set')
    parser.add_argument('--compare-hist', action='store_true', help='eval deleted set')
    parser.add_argument('--eval-deleted', action='store_true', help='eval deleted set')
    parser.add_argument('--num-eval-try', type=int, default=3, help='number of random runs')
    parser.add_argument('--label', type=str, required=True, help='what is the label column for classification')
    parser.add_argument('--test-frac', type=float, default=0.3, help='what is the label column for classification')
    parser.add_argument('--hist-atts', type=str, default='', help='comma separated list of attributes to compare the histograms')
    parser.add_argument('--hist-num-samples', type=int, default=30000, help='number of samples to compare histograms')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs for training or unlearning')
    parser.add_argument('--alpha', type=float, default=0.9, help='alpha for NegGrad+')
    parser.add_argument('--pre-model', type=str, default=None, help='address to the previous model')
    parser.add_argument('--filters', type=str, default=None, help='address to the json file for the rows to be deleted')
    parser.add_argument('--delete-frac', type=float, default=1, help='fraction of the rows to be deleted')

    #Wandb arguments
    parser.add_argument('--wandb-mode', type=str, default='disabled', choices=['online', 'offline', 'disabled'], 
                        help='wandb running mode')
    parser.add_argument('--wandb-project', type=str, default=None,
                        help='the project on wandb to add the runs')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='your wandb user name')
    parser.add_argument('--wandb-run-id', type=str, default=None,
                        help='To resume a previous run with an id')
    parser.add_argument('--wandb-group-name', type=str, default=None,
                        help='Given name to group runs together')

    args = parser.parse_args()
    atts = args.hist_atts.split(',')
    args.hist_atts = list([])
    for a in atts:
        args.hist_atts.append(a)
    else:
        return args


def init_logger(args):

    if args.wandb_group_name is None:
        args.wandb_group_name = args.dataset
    if args.wandb_run_id is not None:
        logger = wandb.init(id=args.wandb_run_id, resume="must")
    else:
        logger = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                   mode=args.wandb_mode, group=args.wandb_group_name, config=args)


    logger.name = f"{args.dataset}_{args.mode}"
    return logger


def unlearn(csv_filename, meta_filename, filters, pre_model, args, logger=None):

    assert os.path.exists(csv_filename)
    assert os.path.exists(meta_filename)
    assert os.path.exists(filters)
    assert os.path.exists(pre_model)


    data, discrete_columns = data_handler.read_csv(csv_filename=csv_filename,meta_filename=meta_filename)
    retained_data, deleted_data = data_handler.delete_data(data, filters, args.delete_frac, args, logger)

    if isinstance(logger, type(wandb.run)):
        logger.name = logger.name + f"_{logger.config['delete-filter']}_{args.delete_frac}"


    tvae = None
    with open(pre_model,'rb') as inp:
        tvae = pickle.load(inp)


    if args.mode == "Retrain":
        t1 = time.time()
        tvae.fit(retained_data, discrete_columns, args.epochs, retrain=True, logger=logger)
        t2 = time.time()
    if args.mode == "Finetune":
        t1 = time.time()
        tvae.finetune(retained_data, discrete_columns, args.epochs, logger=logger)
        t2 = time.time()
    elif args.mode == "Stale":
        t1 = t2 = 0
        pass
    elif args.mode == "NegGrad":
        t1 = time.time()
        tvae.neggrad(retained_data, deleted_data, discrete_columns, epochs=args.epochs, alpha=0, logger=logger)
        t2 = time.time()
    elif args.mode == "NegGrad+":
        t1 = time.time()
        tvae.neggrad(retained_data, deleted_data, discrete_columns, epochs=args.epochs, alpha=args.alpha, logger=logger)
        t2 = time.time()


    print ("Unlearning took {} seconds".format(t2-t1))
    PATH = f'models/{args.dataset}_{args.mode}.pkl'
    with open(PATH, 'wb') as outp:
        pickle.dump(tvae, outp, pickle.HIGHEST_PROTOCOL)
    
    if args.eval:
        benchmarking.classification(modelpath=PATH, datapath=args.datafile.replace('.csv', '_reduced.csv'), 
            eval_deleted=False, deletedfile=None,
            metapath=args.metadatafile, num_try=args.num_eval_try, label=args.label, frac=args.test_frac, logger=logger)
    if args.eval_deleted:
        benchmarking.classification(modelpath=PATH, datapath=args.datafile.replace('.csv', '_reduced.csv'), 
            metapath=args.metadatafile, num_try=args.num_eval_try, label=args.label, 
            eval_deleted=True, deletedfile=args.datafile.replace('.csv', '_deleted.csv'),
            frac=args.test_frac, logger=logger)

    if args.compare_hist:
        benchmarking.compare_histograms(modelpath=PATH, datafile=args.datafile.replace('.csv', '_reduced.csv'), attributes=args.hist_atts,
            mode=args.mode, num_sample=args.hist_num_samples, logger=logger)

 
    if isinstance(logger, type(wandb.run)) and logger is not None:
        logger.save(PATH, policy='end')
        logger.save(filters, policy='end')

def train(csv_filename, meta_filename, args, pre_model=None, logger=None):

    assert os.path.exists(csv_filename)
    assert os.path.exists(meta_filename)

    data, discrete_columns = data_handler.read_csv(csv_filename=csv_filename,meta_filename=meta_filename)


    if pre_model is not None:
        tvae = None
        with open(pre_model,'rb') as inp:
            tvae = pickle.load(inp)
        t1 = time.time()
        tvae.fit(data, discrete_columns, pre_model, epochs=args.epochs, retrain=True, logger=logger)
        t2 = time.time()
    else:
        tvae = TVAESynthesizer(epochs=args.epochs)
        t1 = time.time()
        tvae.fit(data, discrete_columns, pre_model, logger=logger)
        t2 = time.time()


    print ("Training took {} seconds".format(t2-t1))
    PATH = f'models/{args.dataset}_{args.mode}.pkl'
    with open(PATH, 'wb') as outp:
        pickle.dump(tvae, outp, pickle.HIGHEST_PROTOCOL)

    if args.eval:
        benchmarking.classification(modelpath=PATH, datapath=args.datafile, metapath=args.metadatafile, 
            num_try=args.num_eval_try, label=args.label, eval_deleted=args.eval_deleted, frac=args.test_frac, logger=logger)

    if args.compare_hist:
        benchmarking.compare_histograms(modelpath=PATH, datafile=args.datafile, attributes=args.hist_atts,
            mode=args.mode, num_sample=args.hist_num_samples, logger=logger)
    

    if isinstance(logger, type(wandb.run)) and logger is not None:
        logger.save(PATH, policy='end')

if __name__ == "__main__":
    args = parser()
    logger = init_logger(args)

    if args.mode.lower() == "train":
        train(csv_filename=args.datafile, meta_filename=args.metadatafile, args=args, pre_model=args.pre_model, logger=logger)
    else:
        unlearn(csv_filename=args.datafile, meta_filename=args.metadatafile, filters=args.filters, 
                pre_model=args.pre_model, args=args, logger=logger)
