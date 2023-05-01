import scipy
import numpy as np
import random
import pandas as pd
import pandasql as ps
import torch.nn as nn
import torch
import math
import dill
import os
import json
import shutil
from collections import Counter
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue, Lock, Manager
from scipy import integrate,stats
from sklearn.metrics import mean_absolute_error
import seaborn as sns

from train_mdn import DenMDN, MDN
from Dataset import Data
from sqlParser import Parser
import utils

torch.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)

dill._dill._reverse_typemap['ClassType'] = type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lock = Lock()


class QueryBenchmarking:
    """Benchmarking a model with random queries

    Args:
        num_queries (integer): number of queries to generate
        category_att (string): name of the categorical attribute
        range_att (string): name of the range attribute

    """
    def __init__(self, category_att, range_att, num_queries):
        self.num_queries = num_queries
        self.queries = []
        self.category_att = category_att
        self.range_att = range_att
        self.results = {}


    def generate_queries(self, _range, categories, agg):
        """Generating random queries of the following format: 

            Q = SELECT COUNT(*) FROM table WHERE x='cat' AND y BETWEEN lb AND ub

        Args:
            _range (tuple-like): a tuple including min and max for the range attribute
            categories (list-like): a list of all distinct values for the categorical attribute

        Output:
            self.queries (list-like): a list of generated queries
        """
        template = "select {}({}) from {} where {} >= {} and {} <= {} and {} = '{}'"
        i = 0
        while i < self.num_queries:

            i1 = int(np.random.uniform(_range[0], _range[1]))
            i2 = int(np.random.uniform(_range[0], _range[1]))
               
            if abs(i1 - i2) > 1:
                ub = np.max([i1,i2])
                lb = np.min([i1,i2])
                cat = np.random.choice(categories)
                q = template.format(agg, self.range_att, 'XXXTOBEREPLACEDXXX', self.range_att, lb, self.range_att, ub, self.category_att, cat)
                self.queries.append(q)
                i = i+1
       
    def calculate_groundtruth(self, DATADF, num_workers=30):
        """Calculating ground-truths of the self.queries

        Args:
            DATADF (dataframe): a dataframe of the main table. \
                                used by pandasql to calculate the \
                                query answers


        """

        if self.queries==None:
            print ("No query found!")
            return

        q = Queue()
        for i, query in enumerate(self.queries):
            q.put((i,query))

        manager = Manager()
        self.results = manager.dict()
        
        workers = []
        for i in range(num_workers):
            p = Process(target=self.run_psql_query, args=(q, DATADF))
            p.start()
            workers.append(p)
            #q = q.replace('XXXTOBEREPLACEDXXX','DATADF')
            #re = ps.sqldf(q,locals())
            #results[i] = re.iloc[0][0]
        
        for p in workers:
            p.join()
        print ("threads finished")

        return self.results

    def run_psql_query(self, q, DATADF):
        while not q.empty():
            i, query = q.get()
            query = query.replace('XXXTOBEREPLACEDXXX','DATADF')
            re = ps.sqldf(query,locals())
            result = re.iloc[0][0]
            with lock:
                self.results[i] = result

    def relative_error(self,gt, pred):
        """Calculates relative error using the prediction and ground-truths

        Args:
            gt (map-like): a dictionary of the ground-truths. \
                           the keys are the index of the queries \
                           in self.queries list
            pred (map-like): a dictionary of the ground-truths. \
                             the keys are the index of the queries \
                             in self.queries list
        """

        errors = []
        for key in gt.keys():
            y1 = gt[key]
            y2 = pred[key]
            err = abs(y2-y1)/y1
            errors.append(err)
        
        relative_error = np.mean(errors)
        print (relative_error)
        return relative_error


def log_likelihood_test(basemodel, testset, updatemodels, update_testsets):
    models = [basemodel]
    
    for file in np.sort(os.listdir(updatemodels)):
        if file.startswith('update') and file.endswith('.dill'):
            models.append(os.path.join(updatemodels,file))
    
    testfiles = [testset]
    for file in np.sort(os.listdir(update_testsets)):
        if file.startswith('update') and file.endswith('_test.csv'):
            testfiles.append(os.path.join(update_testsets,file))

    if len(models) != len(testfiles):
        print ('number of models and test_sets does not match!')
        return
    else:
        print ("\n\nThe files and their corresponding testfile:\n")
        for i, itm in enumerate(models):
            print (itm,'\t',testfiles[i])


    lls_modelupdated = []
    lls_modelfixed = []
    all_testsets = pd.DataFrame()
    for i, name in enumerate(models):
        m_fix = None
        with open(models[0], 'rb') as d:
            m_fix = dill.load(d)

        
        m_update = None
        with open(name, 'rb') as d:
            m_update = dill.load(d)


        x_values, y_values = m_update.dataset.read_data(testfiles[i], haveheader=True) 
        df = pd.DataFrame.from_dict({**x_values, **y_values})
        all_testsets = pd.concat([all_testsets,df])
        print (all_testsets.shape)
        xs = {}
        ys = {}
        for key in x_values.keys():
            xs[key] = all_testsets[key].tolist()
        for key in y_values.keys():
            ys[key] = all_testsets[key].tolist()
        del x_values
        del y_values

        x_encoded = {}
        for key in xs.keys():
            x_encoded[key] = m_update.dataset.encoders[key].transform(np.asarray(xs[key]).reshape(-1,1)).toarray()
        y_normalized = m_update.dataset.normalize(ys)
        
        tensor_xs = torch.from_numpy(x_encoded[m_update.dataset.x_attributes[0]].astype(np.float32)) 
        y_points = np.asarray(y_normalized[m_update.dataset.y_attributes[0]]).reshape(-1,1)
        tensor_ys = torch.from_numpy(y_points.astype(np.float32))

        # move variables to cuda
        tensor_xs = tensor_xs.to(device)
        tensor_ys = tensor_ys.to(device)
        
        pi, sigma, mu = m_fix.model(tensor_xs)
        ll = utils.LL(pi, sigma, mu, tensor_ys, device)
        lls_modelfixed.append(ll)

        pi, sigma, mu = m_update.model(tensor_xs)
        ll = utils.LL(pi, sigma, mu, tensor_ys, device)
        lls_modelupdated.append(ll)
    

    
    lls_modelfixed = [t.item() for t in lls_modelfixed]
    lls_modelupdated = [t.item() for t in lls_modelupdated]

    print (lls_modelfixed)
    print (lls_modelupdated)

    x = list(range(len(lls_modelfixed)))
    fig, (ax1, ax2) = plt.subplots(2)

    ax1.plot(x, lls_modelupdated, marker='', color='blue', linewidth=4,label='updated')
    ax1.plot(x, lls_modelfixed, marker='', color='olive', linewidth=2,label='fixed')
    ax1.legend()
    ax1.set_xlabel("step")
    ax1.set_ylabel("log likelihood")
    #ax1.set_ylim([0,1])
    
    ax2.plot(x, lls_modelupdated, marker='o', markerfacecolor='blue', markersize=10, color='skyblue', linewidth=4,label='updated')
    ax2.plot(x, lls_modelfixed, marker='', color='olive', linewidth=2,label='fixed')
    ax2.legend()
    ax2.set_xlabel("step")
    ax2.set_ylabel("likelihood")
    #ax2.set_ylim([0,1])


    plt.savefig('ll.png')

def create_the_benchmark(args, dataset, datafile, benchmark_dir, cat_att, range_att, sep, agg, filters_path=None, num_queries=2000):
    """Setting up model evaluation using query benchmarks

    Args:
        dataset (string): the dataset name
        datafile (string): Path to the main data table in csv format
        benchmark_dir (string): Path to the directory where the queries \
                                will be saved
        cat_att (string): name of the categorical attribute
        range_att (string): name of the range attribure
        sep (string): delimiter in the csv file
        filters_path: the path to the json file containing filters to delete


    Output: 
        a set of queries with their ground-truth saved at: \
        benchmark_dir/{dataset}_queries.sql
        benchmark_dir/{dataset}_queries.csv
        benchmark_dir/{dataset}_deleted_rows_queries.sql
        benchmark_dir/{dataset}_deleted_rows_queries.csv
        benchmark_dir/{dataset}_both_rows_queries.sql
        benchmark_dir/{dataset}_both_rows_queries.csv
    """

    query_retained_data(dataset, datafile, benchmark_dir, cat_att, range_att, sep, agg, num_queries)
    if args.mode != "train" and args.eval_deleted == True:
        query_deleted_data(dataset, benchmark_dir, datafile, filters_path, cat_att, range_att, agg, num_queries)
        #query_deleted_and_retained_data(dataset, benchmark_dir, datafile, filters_path, cat_att, range_att, agg, num_queries)

def query_retained_data(dataset, datafile, benchmark_dir, cat_att, range_att, sep, agg, num_queries):
    """Creating queries for the retained data"""
    
    data_handler = Data(datafile, x_attributes=[cat_att], y_attributes=[range_att], sep=',')    
    df = data_handler.read_data(_return="dataframe")

    _range = (df[range_att].min(), df[range_att].max())

    categories = list(set(df[cat_att].tolist()))
    b = QueryBenchmarking(cat_att, range_att, num_queries)


    print ("Start generating random queries ...")
    b.generate_queries(_range=_range, categories=categories, agg=agg)
    
    queries = os.path.join(benchmark_dir, dataset+"_"+agg+'_queries.sql')
    with open(queries, 'w') as f:
        for q in b.queries:
            f.write(q)
            f.write('\n')
    

    print ("Start generating ground-truth ...")
    results = b.calculate_groundtruth(df)

    keys = np.sort(list(results.keys()))
    ground_truth = os.path.join(benchmark_dir, dataset+"_"+agg+'_queries.csv')
    with open(ground_truth, 'w') as g:
        for key in keys:
            g.write(str(results[key]))
            g.write('\n')

def query_deleted_data(dataset, benchmark_dir, datafile, filters_path, cat_att, range_att, agg, num_queries):
    """Creating queries for the deleted data
       Note that when a categorical attribute is deleted \
       the frequency tables are automatically zeroed out \
       therefore there is no need for evaluation.
       So we only look for range attributes that are filtered (deleted)
    """

    data_handler = Data(datafile, x_attributes=[cat_att], y_attributes=[range_att], sep=',')    
    df = data_handler.read_data(_return="dataframe")

    categories = list(set(df[cat_att].tolist()))

    filters = None
    with open(filters_path, 'r') as f:
        filters = json.load(f)
    filters = filters['filters']

    template = "select {}({}) from {} where {} >= {} and {} <= {} and {} = '{}'"
    queries = []
    for i, _filter in enumerate(filters):
        if _filter['att'] == range_att:
            if _filter['type'] == 'equality':
                lb = _filter['val'] 
                ub = _filter['val']
            elif _filter['type'] == "range_full":
                lb = _filter['min_val']
                ub = _filter['max_val']
            else:
                raise ValueError("The filter type is not suitable for deleted queries!")

            for i in range(num_queries):
                i1 = int(np.random.uniform(lb, ub))
                i2 = int(np.random.uniform(lb, ub))

                ub = np.max([i1,i2])
                lb = np.min([i1,i2])

                if lb == ub:
                    ub = lb + 1

                cat = np.random.choice(categories)
                q = template.format(agg, range_att, 'XXXTOBEREPLACEDXXX', range_att, lb, range_att, ub, cat_att, cat)
                queries.append(q)

    if len(queries) == 0:
        print ("could not generate any queries for the deleted data!")
    
    cardinalities = [0]*len(queries)

    queries_path = os.path.join(benchmark_dir, '{}_{}_deleted_rows_query.sql'.format(dataset,agg))
    cards_path = os.path.join(benchmark_dir, '{}_{}_deleted_rows_query.csv'.format(dataset,agg))
    with open(queries_path,'w') as fq, \
         open(cards_path,'w') as fc:
        for i, q in enumerate(queries):
            fq.write(q)
            fq.write('\n')
            fc.write(str(cardinalities[i]))
            fc.write('\n')

def query_deleted_and_retained_data(dataset, benchmark_dir, datafile, filters_path, cat_att, range_att, agg, num_queries):
    """Creating queries that contain both the retained data and the deleted data"""

    data_handler = Data(datafile, x_attributes=[cat_att], y_attributes=[range_att], sep=',')    
    df = data_handler.read_data(_return="dataframe")

    _range = (df[range_att].min(), df[range_att].max())
    categories = list(set(df[cat_att].tolist()))

    filters = None
    with open(filters_path, 'r') as f:
        filters = json.load(f)
    filters = filters['filters']

    template = "select {}({}) from {} where {} >= {} and {} <= {} and {} = '{}'"
    queries = []
    for i, _filter in enumerate(filters):
        if _filter['att'] == range_att:
            if _filter['type'] == "range":
                lb = _filter['min_val']
                ub = _filter['max_val']

                for i in range(num_queries):
                    i1 = int(np.random.uniform(lb, ub))
                    i2 = int(np.random.uniform(_range[0], _range[1]))
                    if abs(i1 - i2) > 2:
                        ub = np.max([i1,i2])
                        lb = np.min([i1,i2])
                        cat = np.random.choice(categories)
                        q = template.format(agg, range_att, 'XXXTOBEREPLACEDXXX', range_att, lb, range_att, ub, cat_att, cat)
                        queries.append(q)

    if len(queries) == 0:
        print ("could not run queries containing both deleted and retained data!")
    
    cardinalities = [0]*len(queries)

    queries_path = os.path.join(benchmark_dir, '{}_{}_both_rows_query.sql'.format(dataset,agg))
    cards_path = os.path.join(benchmark_dir, '{}_{}_both_rows_query.csv'.format(dataset,agg))

    with open(queries_path,'w') as fq, \
         open(cards_path,'w') as fc:
        for i, q in enumerate(queries):
            fq.write(q)
            fq.write('\n')
            fc.write(str(cardinalities[i]))
            fc.write('\n')

def cal_count(probs, step, frequencies):

    sub_areas = probs * step

    integral = np.sum(sub_areas[:, 1:-1], axis=1)

    integral = np.add(integral, sub_areas[:,0]*0.5)
    integral = np.add(integral, sub_areas[:,-1]*0.5)
    
    _count = integral * np.asarray(frequencies[list(frequencies.keys())[0]]).reshape(-1,1)

    return _count[0][0]

def cal_sum(probs, regs, step, frequencies):
    product = np.multiply(probs, regs)

    sub_areas = product * step

    integral = np.sum(sub_areas[:, 1:-1], axis=1)
    integral = np.add(integral, sub_areas[:,0]*0.5)
    integral = np.add(integral, sub_areas[:,-1]*0.5)
    
    _sum = integral * np.asarray(frequencies[list(frequencies.keys())[0]]).reshape(-1,1)

    return _sum[0][0]

def cal_avg(probs, regs, step, frequencies):
    _count = cal_count(probs, step, frequencies)
    _sum = cal_sum(probs, regs, step, frequencies)

    if np.floor(_sum) == 0:
        return 0
    if np.floor(_count) == 0:
        _count = 1

    _avg = _sum/_count
    return _avg

def cal_error(reals, predictions, metric):
    errs = []
    for query_num, key in enumerate(reals.keys()):
        if reals[key]>-1 and predictions[key]>-1:
            if metric == "relative-error":
                err = abs(reals[key]-predictions[key])/(reals[key])
            elif metric == "q-error": 
                err = np.min([reals[key],predictions[key]])/(np.max([reals[key], predictions[key]]))
            elif metric == "absolute-error":
                err = np.abs(reals[key]-predictions[key])
            else:  
                raise ValueError('The metric must be eigther relative-error, q-error, or absolute-error')

            errs.append(err)
            #print (equalities[key],reals[key],predictions[key][0][0], freqs[key], err)
  
    metrics_summary = {"metric": metric, "min": np.min(errs), "max": np.max(errs),
                       "mean": np.mean(errs),
                       "median": np.median(errs),
                       "90th": np.percentile(np.array(errs),90),
                       "95th": np.percentile(np.array(errs),95),
                       "99th": np.percentile(np.array(errs),99)}

    return metrics_summary

def predict(q, M, predictions, integral_points):
    """process function for run_multiproc()."""
    
    while not q.empty():
        i, query = q.get()

        parser = Parser()
        succ, conditions = parser.parse(query, M.dataset.x_attributes + M.dataset.y_attributes)
        if not succ:
            print ("error in {}".format(query))
            return
        

        x_values = {}
        y_values = {}
        lb = ub = step = 0
        for key in conditions.keys():
            if conditions[key].equalities:
                x_values[key] = conditions[key].equalities
            else:
                lb = M.dataset.normalizers[key].min
                ub = M.dataset.normalizers[key].max
                if conditions[key].lb is not None:
                    lb = conditions[key].lb
                if conditions[key].ub is not None:
                    ub = conditions[key].ub
                y_values[key], step = list(np.linspace(lb,ub,integral_points, retstep=True))


        frequencies = {}        
        x_encoded = {}
        for key in x_values.keys():
            x_encoded[key] = M.dataset.encoders[key].transform(np.asarray(x_values[key]).reshape(-1,1)).toarray()
            frequencies[key] = [M.dataset.FTs[key][cat] for cat in x_values[key]]
            
        y_normalized = M.dataset.normalize(y_values)
        probs = M.predict(x_points = x_encoded[list(x_values.keys())[0]] ,y_points=y_normalized[list(y_values.keys())[0]])
        probs = probs / M.dataset.normalizers[list(y_values.keys())[0]].width * 2


        sub_areas = probs * step

        integral = np.sum(sub_areas[:, 1:-1], axis=1)

        integral = np.add(integral, sub_areas[:,0]*0.5)
        integral = np.add(integral, sub_areas[:,-1]*0.5)

        count = integral * np.asarray(frequencies[list(frequencies.keys())[0]]).reshape(-1,1)


        with lock:
            predictions[i] = count

def run_multiproc(model, queries_file, ground_truth, integral_points=100, num_workers=2):
    """Running the benchmark, multi-process

    Args:
        model (string or Obj): Path to the pretrained model or a pre-loaded DenMDN class object
        queries (string): Path to the generated queries
        ground_truth (string): Path to the ground-truth csv file
        integram_points (integer): number of sub-rectangular areas when \
                                   calculating the integral of the density \
        num_workders (integer): number of workers for multiprocessing
    
    """

    M = None
    if type(model) == np.str:
        with open(model, 'rb') as d:
            M = dill.load(d)
    else:
        M = model

    queries = []
    with open(queries_file,'r') as file:
        for line in file:
            qs = line.split(';')
            for itm in qs:
                if itm.lower().startswith('select'):
                    queries.append(itm)
                elif len(itm)>1:
                    print ('query not recognized \n {}'.format(itm))

    reals = {}
    with open(ground_truth,'r') as file:
        for i, line in enumerate(file):
            reals[i] = int(line.split(',')[0])


    q = Queue()
    for i, query in enumerate(queries):
        q.put((i,query))

    manager = Manager()
    predictions = manager.dict()
    
    workers = []
    for i in range(num_workers):
        p = Process(target=predict, args=(q, M, predictions, integral_points))
        p.start()
        workers.append(p)
        #q = q.replace('XXXTOBEREPLACEDXXX','DATADF')
        #re = ps.sqldf(q,locals())
        #results[i] = re.iloc[0][0]
    
    for p in workers:
        p.join()
    print ("threads finished")



    errs = []
    for query_num, key in enumerate(reals.keys()):
        if reals[key]>1 and predictions[key][0][0]>1:
            #err = 100 * abs(reals[key]-predictions[key][0][0])/reals[key]         
            err = np.max([reals[key],predictions[key][0][0]])/np.min([reals[key], predictions[key][0][0]])
            errs.append(err)
            #print (equalities[key],reals[key],predictions[key][0][0], freqs[key], err)
  

    print ("mean q-error = {}, median = {}, 95th={}, 99th={}".format(np.mean(errs),np.median(errs), np.percentile(np.array(errs),95),np.percentile(np.array(errs),99)))

def run(model, queries_file, ground_truth, integral_points=100, metric="absolute-error"):
    """Running the benchmark on a pretrained model
    Args:
        model (string or Obj): Path to the pretrained model or a pre-loaded DenMDN class object
        queries (string): Path to the generated queries
        ground_truth (string): Path to the ground-truth csv file
        integram_points (integer): number of sub-rectangular areas when \
                                   calculating the integral of the density \

    return:
        metrics_summary (map):
            metric: string
            min: float
            max: float
            mean: float
            median: float
            90th percentile: float
            95th percentile: float
            99th percentile: float

    
    """

    M = None
    if type(model) == np.str:
        with open(model, 'rb') as d:
            M = dill.load(d)
    else:
        M = model

    queries = []
    with open(queries_file,'r') as file:
        for line in file:
            qs = line.split(';')
            for itm in qs:
                if itm.lower().startswith('select'):
                    queries.append(itm)
                elif len(itm)>1:
                    print ('query not recognized \n {}'.format(itm))

            
    reals = {}
    with open(ground_truth,'r') as file:
        for i, line in enumerate(file):
            try:
                reals[i] = float(line.split(',')[0])
            except:
                reals[i] = 0


    predictions = {}
    freqs = {}
    equalities = {}
    agg = ""
    for i, query in enumerate(queries):
        parser = Parser()
        succ, conditions, agg = parser.parse(query, M.dataset.x_attributes + M.dataset.y_attributes)
        if not succ:
            print ("error in {}".format(query))
        

        x_values = {}
        y_values = {}
        lb = ub = step = 0
        for key in conditions.keys():
            if conditions[key].equalities:
                x_values[key] = conditions[key].equalities
            else:
                lb = M.dataset.normalizers[key].min
                ub = M.dataset.normalizers[key].max
                if conditions[key].lb is not None:
                    lb = conditions[key].lb
                if conditions[key].ub is not None:
                    ub = conditions[key].ub
                y_values[key], step = list(np.linspace(lb,ub,integral_points, retstep=True))


        frequencies = {}        
        x_encoded = {}
        for key in x_values.keys():
            x_encoded[key] = M.dataset.encoders[key].transform(np.asarray(x_values[key]).reshape(-1,1)).toarray()
            frequencies[key] = [M.dataset.FTs[key][cat] for cat in x_values[key]]                
        y_normalized = M.dataset.normalize(y_values)

        if agg == "count":
            probs= M.predict(x_points = x_encoded[list(x_values.keys())[0]] ,y_points=y_normalized[list(y_values.keys())[0]])
            probs = probs / M.dataset.normalizers[list(y_values.keys())[0]].width * 2
            answer = cal_count(probs, step, frequencies)
            predictions[i] = answer#np.floor(answer[0][0]) if np.floor(answer[0][0])>0 else 1

        elif agg == "sum":
            probs = M.predict(x_points = x_encoded[list(x_values.keys())[0]] ,y_points=y_normalized[list(y_values.keys())[0]])
            probs = probs / M.dataset.normalizers[list(y_values.keys())[0]].width * 2
            regs = y_values[list(y_values.keys())[0]]
            answer = cal_sum(probs, regs, step, frequencies)
            predictions[i] = answer#np.floor(answer[0][0]) if np.floor(answer[0][0])>0 else 0

        elif agg == "avg":
            probs = M.predict(x_points = x_encoded[list(x_values.keys())[0]] ,y_points=y_normalized[list(y_values.keys())[0]])
            probs = probs / M.dataset.normalizers[list(y_values.keys())[0]].width * 2
            regs = y_values[list(y_values.keys())[0]]
            answer = cal_avg(probs, regs, step, frequencies)
            predictions[i] = answer#np.floor(answer[0][0]) if np.floor(answer[0][0])>0 else 0



        freqs[i] =  np.asarray(frequencies[list(frequencies.keys())[0]]).reshape(-1,1)
        equalities[i] = list(x_values.values())[0]



    metrics_summary = cal_error(reals, predictions, metric)

    return metrics_summary, queries, reals, predictions

def plot_histogram(model, datafile, x_att, y_att, x_value, out_name, mode, bins=None):
    data_handler = Data(datafile, x_attributes=[x_att], y_attributes=[y_att], sep=',')    
    df = data_handler.read_data(_return="dataframe")
    p = df[df[x_att]==x_value]

    M = None
    if type(model) == np.str:
        with open(model, 'rb') as d:
            M = dill.load(d)
    else:
        M = model

    x_value = x_value.lower()
    x_encoded = {}
    x_encoded[x_att] = M.dataset.encoders[x_att].transform(np.asarray([x_value]).reshape(-1,1)).toarray()
    tensor_xs = torch.from_numpy(x_encoded[x_att].astype(np.float32))    
    tensor_xs = tensor_xs.to(device)
    pis, sigmas, mus = M.model(tensor_xs)

    softmax = nn.Softmax(dim=1)

    pis = softmax(pis)
    pis = pis.cpu()
    sigmas = sigmas.cpu()
    mus = mus.cpu()

    if bins == None:
        bins = p[y_att].nunique()

    samples = utils.MoGـsampling(pis, sigmas,mus,len(p),device)
    results = [utils.denormalize(i.item(), M.dataset.normalizers[y_att].mean, M.dataset.normalizers[y_att].width) for i in samples[0]]
    q = pd.DataFrame(data={'y':results})



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

    plt.clf()
    color = 'greenish'
    b = sns.histplot(data=p[y_att], color=sns.xkcd_rgb[color], stat='percent')
    b.tick_params(labelsize=16)
    b.set_title(title_real, size=20)
    b.set_xlabel(y_att, size=16)
    b.set_ylabel('percent', size=16)
    path_real = f'results/{out_name}_real.png'
    plt.savefig(path_real)

    plt.clf()
    color = 'bluish'
    b = sns.histplot(data=q['y'], color=sns.xkcd_rgb[color], stat='percent')
    b.tick_params(labelsize=16)
    b.set_title(title, size=20)
    b.set_xlabel(y_att, size=16)
    b.set_ylabel('percent', size=16)
    path_synth = f'results/{out_name}_synthetic.png'
    plt.savefig(path_synth)   

    
    return p[y_att].values, q.y.values, path_real, path_synth

if __name__=="__main__":
    #run(model="retrain01.dill",queries_file='benchmark/alldata/count01.sql', ground_truth='benchmark/alldata/count01.csv', metric='q-error')
    plot("models/census_train_native_country_age_seed1234.dill", '../datasets/census/census.csv', "native_country", "age", " United-States", "census_train", None)
