import wandb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import scipy.stats as st
from pathlib import Path
import os
plt.rcParams["font.family"] = "Times New Roman"



def seq_figures():
    api = wandb.Api()
    entity, project = "USERNAME", "PROJECT"  # set to your entity and project
    runs = api.runs(entity+"/"+project)

    dataset = "census"

    neggrad_runs = ['']


    metrics = ['final-mean', 'final-median', 'final-95th-percentile']
    if dataset == "census":
        forget_sizes = ["2603", "3856", "5191", "6494", "7831"] #census
        methods = ['retrain', 'stale', 'finetune', 'negatgrad+', 'negatgrad', 'scrub']
        original_run = ""
    elif dataset == "forest":
        forget_sizes = ["1273", "9984", "17206", "28367", "41204"] #forest
        methods = ['retrain', 'stale', 'finetune', 'negatgrad+', 'negatgrad', 'scrub']
        original_run = ""
    


    errors_mean_cnt = {m:[None]*(len(forget_sizes)) for m in methods}
    errors_median_cnt = {m:[None]*(len(forget_sizes)) for m in methods}
    errors_95th_cnt = {m:[None]*(len(forget_sizes)) for m in methods}



    for method in methods:
        for i, fsize in enumerate(forget_sizes):
            for run in runs:
                #print (run.config['mode'].lower(), method.lower(), run.config['deleted size'], int(fsize))
                if run.config['mode'].lower() != 'train' and run.config['dataset'] == dataset:
                    names = run.name.replace('_','-').split('-')
                    if names[1].lower() == method.lower() and \
                        run.config['deleted size'] == int(fsize) and \
                        run.config['wandb_group_name'].startswith(f'{dataset}-req'):

                        results = run.summary

                        #errors 
                        errors_mean_cnt[method][i] = results['final-mean']*100
                        errors_median_cnt[method][i] = results['final-median']*100
                        errors_95th_cnt[method][i] = results['final-95th-percentile']*100

                    

    """
    #Original
    original = api.run(original_run)
    results = original.summary

    for method in methods:
        errors_mean_cnt[method] = [results['final-mean']*100] + errors_mean_cnt[method]
        errors_median_cnt[method] = [results['final-median']*100] + errors_median_cnt[method]
        errors_95th_cnt[method] = [results['final-95th-percentile']*100] + errors_95th_cnt[method]
    """

    # Add data
    palette = plt.get_cmap('Set1')
    markers = ['.', '^', '*', 'x', '+', '4']

    errors_r = [errors_mean_cnt, errors_median_cnt, errors_95th_cnt]

    titles_r = ['Count Queries On The Retain Rows', 'Count Queries On The Retain Rows', 'Count Queries On The Retain Rows']
    y_labels_r = ['Mean Relative Error', 'Median Relative Error', '95th Relative Error']


    #Retain errors
    labels = ['req1', 'req2', 'req3', 'req4', 'req5']
    for j, error in enumerate(errors_r):
        for i, method in enumerate(methods):
            if method == 'scrub':
                tag = 'SCRUB'
            elif method == 'negatgrad+':
                tag = 'NegGrad+'
            elif method == 'negatgrad':
                tag = 'NegGrad'
            elif method == 'retrain':
                tag = 'Retrain'
            elif method == 'finetune':
                tag = 'Fine-tune'
            elif method == 'stale':
                tag = 'Stale'

            print (method, error[method])
            plt.plot(labels, error[method], marker=markers[i], color=palette(i), linewidth=2, alpha=1, label=tag)



        plt.legend(loc=2, ncol=2)
         

        plt.title(dataset, loc='center', fontsize=20, fontweight=4, color='black')
        plt.xlabel("delete request", fontsize=18)
        plt.ylabel(y_labels_r[j], fontsize=18)
        plt.xticks(fontsize=16)
        # after plotting the data, format the labels
        current_values = plt.gca().get_yticks()
        if (current_values >= 1000).any():
            #plt.gca().yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

        
        Path(f"results/").mkdir(parents=True, exist_ok=True)
        name = f"results/naru_sequential_{dataset}_{titles_r[j].replace(' ','_')}_{y_labels_r[j].replace(' ', '_')}.png"
        plt.savefig(name)

        plt.clf()

def ratio_figures():

    api = wandb.Api()
    entity, project = "USERNAME", "PROJECT"  # set to your entity and project

    dataset = "forest"

    if dataset == "census":
        #census_oneshot
        runs_oneshot = []

        #census_seq
        runs_seq = []

    elif dataset == "forest":
        #forest_oneshot
        runs_oneshot = []

        #forest_seq
        runs_seq = []


    style_names = ['mean-cnt', 'median-cnt', '95th-cnt']
    errors = {}

    for r_j, r_l in zip(runs_oneshot, runs_seq):
        run_j = api.run(r_j)
        run_l = api.run(r_l)
        names_j = run_j.name.replace('-','_').split("_")
        names_l = run_l.name.replace('-','_').split("_")
        assert names_j[1] == names_l[1]
        method = names_j[1]

        errs = {}
        for case, run in zip(["oneshot", "seq"],[run_j, run_l]):
            results = run.summary
            #errors 

            errs[case] = {"count": [results["final-mean"], results["final-median"], results["final-95th-percentile"]]}


        errors[method] = errs



    #Creating ratio figures
    palette = plt.get_cmap('Set1')
    markers = ['.', '^', '*', 'x', '+', '4']

    methods = list(errors.keys())
    for i, method in enumerate(methods):
        if method == 'scrubs':
            methods[i] = 'SCRUB'
        elif method == 'negatgrad+':
            methods[i] = 'NegGrad+'
        elif method == 'negatgrad':
            methods[i] = 'NegGrad'
        elif method == 'retrain':
            methods[i] = 'Retrain'
        elif method == 'finetune':
            methods[i] = 'Fine-tune'
        elif method == 'stale':
            methods[i] = 'Stale'


    methods = tuple(methods)
    labels = ['mean-cnt', 'median-cnt', '95th-cnt']

    #retain errors
    mean_cnt = [errors[method]["oneshot"]["count"][0]/errors[method]["seq"]["count"][0] for method in errors.keys()]
    med_cnt = [errors[method]["oneshot"]["count"][1]/errors[method]["seq"]["count"][1] for method in errors.keys()]
    th_cnt = [errors[method]["oneshot"]["count"][2]/errors[method]["seq"]["count"][2] for method in errors.keys()]


    errs_r = [mean_cnt, med_cnt, th_cnt]
    errors_r = {label:err for label,err in zip(labels,errs_r)}



    #Figures
    x = np.arange(len(methods))  # the label locations
    width = 0.14  # the width of the bars
    multiplier = 0
    palette = plt.get_cmap('Set1')

    fig, ax = plt.subplots(constrained_layout=True)
    ax.tick_params(axis='both', which='major', labelsize=14)

    for i, (metric, errs) in enumerate(errors_r.items()):
        offset = width * multiplier
        rects = ax.bar(x + offset, errs, width, label=metric)
        #ax.bar_label(rects, padding=2)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('one-go/sequential error ratio', fontsize=14, fontweight=4)
    #ax.set_title('the ratio of oneshot deletion over sequential deletion for different metrics', fontsize=14, fontweight=4, color='black')
    ax.set_xticks(x+0.2)
    ax.set_xticklabels(methods, rotation=45, ha='right', rotation_mode='anchor')
    ax.legend(loc='upper left', ncol=3, fontsize=14)
    #ax.plot(methods, [1]*(len(methods)), "k--")
    ax.axhline(y = 1, color = 'black', linestyle = 'dashed')
    #ax.grid()
    ax.set_ylim(0, 2)


    Path("results").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"results/naru_retained_{dataset}_ratios.png")

    plt.clf()

def tables():
    api = wandb.Api()
    entity, project = "USERNAME", "PROJECT"  # set to your entity and project
    runs = api.runs(entity+"/"+project)

    dataset = "forest"
    group = f'{dataset}-compare-last-points-v1'

    if dataset == "census":
        forget_sizes = ["3915", "7831"] #census
        methods = ['train', 'retrain', 'stale', 'finetune', 'negatgrad+', 'negatgrad', 'scrub']

    elif dataset == "forest":
        forget_sizes = ["53359", "26679"] #forest
        methods = ['train','retrain', 'stale', 'finetune', 'negatgrad+', 'negatgrad', 'scrub']

    elif dataset == "dmv":
        forget_sizes = ["164508", "82254"] #dmv 
        methods = ['train','retrain', 'stale', 'finetune', 'negatgrad+', 'negatgrad', 'scrub']


    for forget_size in forget_sizes:
        print ("forget size: ", forget_size)
        for run in runs:
            if run.config['dataset'] == dataset and run.group == group:
                if run.name.split('_')[-1].endswith(forget_size) or run.config['mode'] == 'train':


                    if os.path.exists("results_tosave.csv"):
                        os.remove("results_tosave.csv")

                    wandb.restore("results_tosave.csv", run_path="/".join(run.path))

                    df = pd.read_csv('results_tosave.csv')

                    mean_ = np.mean(df['err'])*100
                    median_ = np.median(df['err'])*100
                    percentile_ = np.percentile(df['err'], 95)*100

                    row = run.config['mode'] + '&'
                    row = row + "{:.2f}".format(mean_) + "&" + "{:.2f}".format(median_) + "&" + "{:.2f}".format(percentile_)
                    row = row + "\\\\"
                    print (row)

def confidence_intervals():
    api = wandb.Api()
    entity, project = "USERNAME", "PROJECT"  # set to your entity and project
    runs = api.runs(entity+"/"+project)

    dataset = "dmv"

    if dataset == "census":
        forget_sizes = ["3915", "7831"] #census
        methods = ['train', 'retrain', 'stale', 'finetune', 'negatgrad+', 'negatgrad', 'scrub']

    elif dataset == "forest":
        forget_sizes = ["53359", "26679"] #forest
        methods = ['train','retrain', 'stale', 'finetune', 'negatgrad+', 'negatgrad', 'scrub']

    elif dataset == "dmv":
        forget_sizes = ["164508", "82254"] #dmv
        methods = ['train','retrain', 'stale', 'finetune', 'negatgrad+', 'negatgrad', 'scrub']

    for forget_size in forget_sizes:
        print ("forget size: ", forget_size)
        for run in runs:
            if run.config['dataset'] == dataset and run.group == f'{dataset}-v1':
                if run.name.split('_')[-1].endswith(forget_size) or run.config['mode'] == 'train':


                    if os.path.exists("results_tosave.csv"):
                        os.remove("results_tosave.csv")

                    wandb.restore("results_tosave.csv", run_path="/".join(run.path))

                    df = pd.read_csv('results_tosave.csv')

                    errs = df['err']*100

                    interval = st.t.interval(alpha=0.95, df=len(errs)-1, loc=np.mean(errs), scale=st.sem(errs))

                    row = run.config['mode'] + '&'
                    row = row + "{:.2f}".format(np.mean(interval)) + "±" + "{:.2f}".format(interval[1]-np.mean(interval))
                    row = row + "\\\\"
                    print (row)


if __name__ == "__main__":
    #seq_figures()
    ratio_figures()
    #confidence_intervals()
    #tables()


