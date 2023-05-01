import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import wandb
import time, os, copy, argparse
import pickle, pandas as pd, numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import scipy.stats as st

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import cross_entropy

from ctgan import TVAESynthesizer
from ctgan import load_demo
from ctgan import data


pd.options.mode.chained_assignment = None
from warnings import simplefilter
simplefilter(action='ignore', category=(pd.errors.PerformanceWarning))

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=[
     'classification', 'histograms'],
      help='the evaluation method?')
    parser.add_argument('--dataset', type=str, required=True, choices=[
     'census', 'forest', 'dmv'],
      help='Dataset.')
    parser.add_argument('--datafile', type=str, help='path to the csv datafile', required=True)
    parser.add_argument('--deletedfile', type=str, help='path to the csv deleted file')
    parser.add_argument('--metadatafile', type=str, help='path to the meta data json', required=True)
    parser.add_argument('--model', type=str, help='path to the model to evaluate')
    parser.add_argument('--eval-deleted', action='store_true', help='eval deleted set')
    parser.add_argument('--num-eval-try', type=int, default=3, help='number of random runs')
    parser.add_argument('--label', type=str, required=True, help='what is the label column for classification')
    parser.add_argument('--test-frac', type=float, default=0.3, help='what is the label column for classification')
    parser.add_argument('--hist-atts', type=str, default='', help='comma separated list of attributes to compare the histograms')
    parser.add_argument('--hist-num-samples', type=int, default=30000, help='number of samples to compare histograms')
    args = parser.parse_args()
    atts = args.hist_atts.split(',')
    args.hist_atts = list([])
    for a in atts:
        args.hist_atts.append(a)
    else:
        return args


def prepare_classification_data(real, synthetic, target, cat_cols, num_cols, deleted, frac=0.2):
    encoded_df = None
    encoded_synth = None
    normalized_df = None
    normalized_synth = None
    if len(cat_cols) > 0:
        encoder = ce.BinaryEncoder(cols=cat_cols, return_df=True)
        encoded_df = encoder.fit_transform(real[cat_cols])
        encoded_synth = encoder.transform(synthetic[cat_cols])
    if len(num_cols) > 0:
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(real[num_cols])
        x_scaled_synth = min_max_scaler.transform(synthetic[num_cols])
        normalized_df = pd.DataFrame(x_scaled, columns=num_cols)
        normalized_synth = pd.DataFrame(x_scaled_synth, columns=num_cols)

    if encoded_df is not None and normalized_df is not None:
        X = pd.concat([encoded_df, normalized_df], axis=1)
        X_synth = pd.concat([encoded_synth, normalized_synth], axis=1)
    elif encoded_df is None:
        X = normalized_df
        X_synth = normalized_synth
    elif normalized_df is None:
        X = encoded_df
        X_synth = encoded_synth


    enc = LabelEncoder()
    target_encoded = enc.fit_transform(real[target].values)
    target_encoded_synth = enc.transform(synthetic[target].values)
    
    if deleted is not None:
        if encoded_df is not None and normalized_df is not None:
            encoded_deleted = encoder.transform(deleted[cat_cols])
            X_deleted = encoded_deleted
            x_scaled_deleted = min_max_scaler.transform(deleted[num_cols])
            normalized_deleted = pd.DataFrame(x_scaled_deleted, columns=num_cols)
            X_deleted = pd.concat([encoded_deleted, normalized_deleted], axis=1)
        elif normalized_df is None:
            encoded_deleted = encoder.transform(deleted[cat_cols])
            X_deleted = encoded_deleted
        elif encoded_df is None:
            x_scaled_deleted = min_max_scaler.transform(deleted[num_cols])
            normalized_deleted = pd.DataFrame(x_scaled_deleted, columns=num_cols)
            X_deleted = normalized_deleted

        target_encoded_deleted = enc.transform(deleted[target].values)
        X_train, y_train, X_test, y_test = X.values, target_encoded, X_deleted.values, target_encoded_deleted
    else:
        X_train, X_test, y_train, y_test = train_test_split((X.values), target_encoded, test_size=frac)

    
    return (X_train, y_train), (X_test, y_test), (X_synth.values, target_encoded_synth)


def classification(modelpath, datapath, metapath, num_try, label, eval_deleted, deletedfile=None, frac=0.2, logger=None):
    df, cat_cols = data.read_csv(csv_filename=(datapath), meta_filename=(metapath))
    num_cols = [col for col in df.columns if col not in cat_cols]
    cat_cols.remove(label)
    df_deleted = None
    if eval_deleted:
        df_deleted, _ = data.read_csv(csv_filename=deletedfile, meta_filename=metapath)

    real_data_acc = {'adaboost':[],  'xgboost':[]}
    real_data_f1 = {'adaboost':[],  'xgboost':[]}
    syn_data_acc = {'adaboost':[],  'xgboost':[]}
    syn_data_f1 = {'adaboost':[],  'xgboost':[]}
    real_data_rocauc = {'adaboost':[],  'xgboost':[]}
    syn_data_rocauc = {'adaboost':[],  'xgboost':[]}
    for i in range(num_try):
        df = df.sample(frac=1)
        synthetic = sample(modelpath, int(len(df) * (1 - frac)))
        (X_train, y_train), (X_test, y_test), (X_synth, y_synth) = prepare_classification_data(df, synthetic, label, cat_cols, num_cols, df_deleted, frac=frac)
        clf = XGBClassifier()
        clf.fit(X_train, y_train)
        y_pred_xgboost = clf.predict(X_test)
        y_pred_xgboost = [round(value) for value in y_pred_xgboost]
        f1_xg = f1_score(y_test, y_pred_xgboost, average='macro')
        real_data_f1['xgboost'].append(f1_xg)
        missed = []
        for itm in set(y_train):
            if itm not in set(y_synth):
                missed.append(itm)
        print('classes missed in the synthetic data', missed)
        for itm in missed:
            y_synth = np.append(y_synth, missed)
            X_synth = np.append(X_synth, np.zeros_like(X_synth[0])).reshape(X_synth.shape[0] + 1, X_synth.shape[1])

        clf = XGBClassifier()
        clf.fit(X_synth, y_synth)
        y_pred_xgboost = clf.predict(X_test)
        y_pred_xgboost = [round(value) for value in y_pred_xgboost]
        f1_xg = f1_score(y_test, y_pred_xgboost, average='macro')
        syn_data_f1['xgboost'].append(f1_xg)

    real_data_interval = st.t.interval(alpha=0.95, df=len(real_data_f1['xgboost'])-1, loc=np.mean(real_data_f1['xgboost']), scale=st.sem(real_data_f1['xgboost']))
    syn_data_interval = st.t.interval(alpha=0.95, df=len(syn_data_f1['xgboost'])-1, loc=np.mean(syn_data_f1['xgboost']), scale=st.sem(syn_data_f1['xgboost']))

    if logger is not None:
        if eval_deleted:
            wandb.log({'average f1 deleted real':np.mean(real_data_f1['xgboost']), 
                        'std f1 deleted real':np.std(real_data_f1['xgboost']), 
                        'average f1 deleted synth':np.mean(syn_data_f1['xgboost']), 
                        'std f1 deleted synth':np.std(syn_data_f1['xgboost']),
                        'f1 lower_ci deleted real': real_data_interval[0],
                        'f1 upper_ci deleted real': real_data_interval[1],
                        'f1 lower_ci deleted synth': syn_data_interval[0],
                        'f1 upper_ci deleted synthg': syn_data_interval[1]
                        })

            wandb.log({'deleted f1 real': real_data_f1['xgboost'], 'deleted f1 synth': syn_data_f1['xgboost']})
        else:
            wandb.log({'average f1 real':np.mean(real_data_f1['xgboost']),  
                        'std f1 real':np.std(real_data_f1['xgboost']), 
                        'average f1 synth':np.mean(syn_data_f1['xgboost']), 
                        'std f1 synth':np.std(syn_data_f1['xgboost']),
                        'f1 lower_ci real': real_data_interval[0],
                        'f1 upper_ci real': real_data_interval[1],
                        'f1 lower_ci synth': syn_data_interval[0],
                        'f1 upper_ci synthg': syn_data_interval[1]
                        })

            wandb.log({'f1 real': real_data_f1['xgboost'], 'f1 synth': syn_data_f1['xgboost']})
    elif eval_deleted:
        print('f1 deleted score  \t\t\t{:.4f},{:.4f}\t\t{:.4f},{:.4f}'.format(np.mean(real_data_f1['xgboost']), 
                        np.std(real_data_f1['xgboost']), np.mean(syn_data_f1['xgboost']), np.std(syn_data_f1['xgboost'])))
    else:
        print('f1 score  \t\t\t{:.4f},{:.4f}\t\t{:.4f},{:.4f}'.format(np.mean(real_data_f1['xgboost']), 
                        np.std(real_data_f1['xgboost']), np.mean(syn_data_f1['xgboost']), np.std(syn_data_f1['xgboost'])))
    return (real_data_f1, syn_data_f1)


def sample(modelname, num_sample):
    model = None
    with open(modelname, 'rb') as (inp):
        model = pickle.load(inp)
    if model:
        sample = model.sample(num_sample)
        return sample
    print('failed loading the model!')
    return None


def compare_histograms(modelpath, datafile, attributes, mode, plot_type='hist', num_sample=30000, logger=None):
    fsample = sample(modelpath, num_sample)
    if fsample is None:
        return
    d, _ = data.read_csv(datafile)
    rsample = d.sample(min(num_sample, len(d) - 1))

    title = ""
    if mode.lower() == "train":
        title = 'Train'
        title_real = 'Original data'
    elif mode.lower() == "retrain":
        title = 'Retrain'
        title_real = 'Original data after deletion'
    elif mode.lower() == "finetune":
        title = 'Fine-tune'
        title_real = title
    elif mode.lower() == "stale":
        title = 'Stale'
        title_real = title
    elif mode.lower() == "sgda":
        title = 'SCRUB'
        title_real = title
    elif mode.lower() == "neggrad":
        title = 'NegGrad'
        title_real = title
    elif mode.lower() == "neggrad+":
        title = 'NegGrad+'
        title_real = title
    else:
        title = mode
        title_real = title

    plt.tick_params(labelsize=12)
    i = 0
    for att in attributes:

        #real data
        bins = rsample[att].nunique()

        plt.clf()
        color = 'greenish'
        if plot_type == 'hist':
            b = sns.histplot(data=rsample[att], color=sns.xkcd_rgb[color], stat='percent')
            b.set_title(title_real, size=20)
            b.set_xlabel(att, size=16)
            b.set_ylabel('percent', size=16)
            b.tick_params(labelsize=16)
        elif plot_type == 'cat':
            b = sns.catplot(data=rsample, x=att, kind="count", color=sns.xkcd_rgb[color])
            b.set_titles(title)
            b.set_xticklabels(rotation=45)
        else:
            raise ValueError("plot type is not define!")
        


        out_name = modelpath.split('/')[-1].split('.')[0]
        path = 'results/' + out_name + '_real.png'
        plt.savefig(path)
        if isinstance(logger, type(wandb.run)) and logger is not None:
            logger.save(path, policy='end')

        #synthetic samples
        bins = fsample[att].nunique()

        plt.clf()
        color = 'bluish'
        if plot_type == 'hist':
            b = sns.histplot(data=fsample[att], color=sns.xkcd_rgb[color], stat='percent')
            b.set_title(title, size=20)
            b.set_xlabel(att, size=16)
            b.set_ylabel('percent', size=16)
            b.tick_params(labelsize=16)
        elif plot_type == 'cat':
            b = sns.catplot(data=fsample, x=att, kind="count", color=sns.xkcd_rgb[color])
            b.set_titles(title)
            b.set_xticklabels(rotation=45)

        path = 'results/' + out_name + '_synthetic.png'
        plt.savefig(path)
        if isinstance(logger, type(wandb.run)) and logger is not None:
            logger.save(path, policy='end')


if __name__ == '__main__':
    args = parser()
    if args.mode == 'classification':
        classification(modelpath=(args.model), datapath=(args.datafile), metapath=(args.metadatafile), num_try=(args.num_eval_try), label=(args.label), eval_deleted=(args.eval_deleted), frac=(args.test_frac), args=args)
    elif args.mode == 'histograms':
        compare_histograms(args.model, args.dataset, args.datafile, args.hist_atts, args.hist_num_samples, args)
