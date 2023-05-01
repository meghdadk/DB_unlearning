import wandb
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st

import benchmarking
import Dataset
from mdn import MDN, DenMDN
plt.rcParams["font.family"] = "Times New Roman"


def cal_error(reals, predictions, metric):
    assert len(reals) == len(predictions)
    errs = []
    for query_num, key in enumerate(reals.keys()):
        if reals[key]>-1 and predictions[key]>-1: #sometimes needed to ignore outliers
            if metric == "relative-error":
                try:
                    err = abs(reals[key]-predictions[key])/(reals[key])*100
                    errs.append(err)
                except:
                    #err = (abs(reals[key]-predictions[key])+1e-3)/(reals[key]+1e-3)*100
                    #errs.append(err)
                    #print ("passed query number {} for value {}".format(query_num, reals[key]))
                    pass
            elif metric == "q-error":
                try:
                    err = np.min([reals[key],predictions[key]])/(np.max([reals[key], predictions[key]]))
                    errs.append(err)
                except:
                    pass
            elif metric == "absolute-error":
                err = np.abs(reals[key]-predictions[key])
                errs.append(err)
            else:  
                raise ValueError('The metric must be eigther relative-error, q-error, or absolute-error')
 
    interval = st.t.interval(alpha=0.95, df=len(errs)-1, loc=np.mean(errs), scale=st.sem(errs))
    metrics_summary = {"metric": metric, "min": np.min(errs), "max": np.max(errs),
                       "mean": np.mean(errs),
                       "median": np.median(errs),
                       "90th": np.percentile(np.array(errs),90),
                       "95th": np.percentile(np.array(errs),95),
                       "99th": np.percentile(np.array(errs),99),
                       "lower_ci": interval[0],
                       "upper_ci": interval[1],
                       "ci_mean": np.mean(interval)}

    return metrics_summary


def tables():
    api = wandb.Api()
    entity, project = "USERNAME", "PROJECT"  # set to your entity and project
    #runs = api.runs(entity+"/"+project)

    dataset = "dmv"
    selective_full = "selective"

    if dataset == "census":
        if selective_full == "full":
            runs = []


        elif selective_full == "selective":
            runs = []

    elif dataset == "forest":
        if selective_full == "full":
            runs = []

        elif selective_full == "selective":
            runs = []

    elif dataset == "dmv":
        if selective_full == "full":
            runs = []

        elif selective_full == "selective":
            runs = []


    print ("results for {}".format(dataset))

    metrics_likelihood = ['final-del-likelihood']
    style_names = ['mean-cnt', 'median-cnt', '95th-cnt', 'mean-sum', 'median-sum', '95th-sum']

    #print ("&".join(style_names*2+['likelihood']))
    for r in runs:
        run = api.run(r)
        names = run.name.split("_")
        results = run.summary

        #errors 
        cnt_reals = run.config['ground-truth-count']
        cnt_preds = run.config['predictions-count']

        sum_reals = run.config['ground-truth-sum']
        sum_preds = run.config['predictions-sum']

        cnt_errors = cal_error(cnt_reals, cnt_preds, 'relative-error')
        sum_errors = cal_error(sum_reals, sum_preds, 'relative-error')

        if names[1] != 'train':
            try:
                cnt_deleted_reals = run.config['ground-truth-deleted-count']
                cnt_deleted_preds = run.config['predictions-deleted-count']

                sum_deleted_reals = run.config['ground-truth-deleted-sum']
                sum_deleted_preds = run.config['predictions-deleted-sum']

                cnt_deleted_errors = cal_error(cnt_deleted_reals, cnt_deleted_preds, 'absolute-error')
                sum_deleted_errors = cal_error(sum_deleted_reals, sum_deleted_preds, 'absolute-error')
            except:
                print ("deleted workload failed! probably 'selective' mode")

        #print order
        row = run.config['mode'] + " & "
        row += "{:.2f}".format(cnt_errors["mean"]) + " & " + "{:.2f}".format(cnt_errors["median"]) + " & " + "{:.2f}".format(cnt_errors["95th"]) + " & "
        row += "{:.2f}".format(sum_errors["mean"]) + " & " + "{:.2f}".format(sum_errors["median"]) + " & " + "{:.2f}".format(sum_errors["95th"]) + " & "

        if names[1] != 'train':
            try:
                row += "{:.2f}".format(cnt_deleted_errors["mean"]) + " & " + "{:.2f}".format(cnt_deleted_errors["median"]) + " & " + "{:.2f}".format(cnt_deleted_errors["95th"]) + " & "
                row += "{:.2f}".format(sum_deleted_errors["mean"]) + " & " + "{:.2f}".format(sum_deleted_errors["median"]) + " & " + "{:.2f}".format(sum_deleted_errors["95th"]) + " & "
            except:
                pass

        if names[1] != 'train':
            try:
                row +=  "{:.2f}".format(results['final-retain-likelihood']) + " & " + "{:.2f}".format(results['final-del-likelihood'])
            except:
                pass

        row += " \\\\"
        #print (row)


        #errors for the deleted errors
        #for met in metrics_deleted+metrics_likelihood:
        #    try:
        #        row += " & " + "{:.2f}".format(results[met])
        #    except:
        #        #print (met)
        #        pass
        #row += " \\\\"
        row1 = "&" + run.config['mode'] + " & "
        row1 = row1 + "{:.2f}".format(cnt_errors["ci_mean"]) + "±" + "{:.2f}".format(cnt_errors["upper_ci"]-cnt_errors["ci_mean"]) + " & "
        row1 = row1 + "{:.2f}".format(sum_errors["ci_mean"]) + "±" + "{:.2f}".format(sum_errors["upper_ci"]-sum_errors["ci_mean"]) + " & "
        if names[1] != "train":
            try:
                row1 = row1 + "{:.2f}".format(cnt_deleted_errors["ci_mean"]) + "±" + "{:.2f}".format(cnt_deleted_errors["upper_ci"]-cnt_deleted_errors["ci_mean"]) + " & "
                row1 = row1 + "{:.2f}".format(sum_deleted_errors["ci_mean"]) + "±" + "{:.2f}".format(sum_deleted_errors["upper_ci"]-sum_deleted_errors["ci_mean"])
            except:
                pass
        row1 += " \\\\"      
        print (row1)

def seq_figures():
    api = wandb.Api()
    entity, project = "USERNAME", "PROJECT"  # set to your entity and project
    runs = api.runs(entity+"/"+project)


    metric_likelihood = ['final-del-likelihood']

    dataset = 'forest'

    if dataset == "census":
        forget_sizes = ["2603", "2588", "2640", "2628", "2470"] #census
        original_run = "" #census
        methods = ['Retrain', 'Stale', 'Finetune', 'NegGrad', 'NegGrad+', 'SCRUB']
    elif dataset == "forest":
        forget_sizes = ["7593", "12093", "11477", "8842", "13354"] #forest
        original_run = "" #forest
        methods = ['Retrain', 'Stale', 'Finetune', 'NegGrad', 'NegGrad+', 'SCRUB']
    elif dataset == "dmv":
        forget_sizes = ["106661", "30260", "27240", "50936", "45527"]
        original_run = ""
        methods = ['Retrain', 'Stale', 'Finetune', 'NegGrad', 'NegGrad+', 'SCRUB']

    print ("results for {}".format(dataset))
    errors_mean_sum = {m:[None]*(len(forget_sizes)) for m in methods}
    errors_median_sum = {m:[None]*(len(forget_sizes)) for m in methods}
    errors_95th_sum = {m:[None]*(len(forget_sizes)) for m in methods}
    errors_mean_cnt = {m:[None]*(len(forget_sizes)) for m in methods}
    errors_median_cnt = {m:[None]*(len(forget_sizes)) for m in methods}
    errors_95th_cnt = {m:[None]*(len(forget_sizes)) for m in methods}

    errors_mean_sum_del = {m:[None]*(len(forget_sizes)) for m in methods}
    errors_median_sum_del = {m:[None]*(len(forget_sizes)) for m in methods}
    errors_95th_sum_del = {m:[None]*(len(forget_sizes)) for m in methods}
    errors_mean_cnt_del = {m:[None]*(len(forget_sizes)) for m in methods}
    errors_median_cnt_del = {m:[None]*(len(forget_sizes)) for m in methods}
    errors_95th_cnt_del = {m:[None]*(len(forget_sizes)) for m in methods}

    likelihood = {m:[None]*(len(forget_sizes)) for m in methods}

    
    for method in methods:
        for i, fsize in enumerate(forget_sizes):
            for run in runs:
                name = run.name.split('_')
                if method.lower() == name[1].lower() and f"lendel{fsize}" == name[-1] and run.config['dataset'].lower() == dataset and run.group == f"{dataset}-req{i+1}":
                    results = run.summary
                    print (method, run.group)
                    likelihood[method][i] = results[metric_likelihood[0]]

                    #errors 
                    cnt_reals = run.config['ground-truth-count']
                    cnt_preds = run.config['predictions-count']

                    sum_reals = run.config['ground-truth-sum']
                    sum_preds = run.config['predictions-sum']

                    cnt_errors = cal_error(cnt_reals, cnt_preds, 'relative-error')
                    sum_errors = cal_error(sum_reals, sum_preds, 'relative-error')

                    if name[1] != 'train':

                        cnt_deleted_reals = run.config['ground-truth-deleted-count']
                        cnt_deleted_preds = run.config['predictions-deleted-count']

                        sum_deleted_reals = run.config['ground-truth-deleted-sum']
                        sum_deleted_preds = run.config['predictions-deleted-sum']

                        cnt_deleted_errors = cal_error(cnt_deleted_reals, cnt_deleted_preds, 'absolute-error')
                        sum_deleted_errors = cal_error(sum_deleted_reals, sum_deleted_preds, 'absolute-error')



                    errors_mean_cnt[method][i] = cnt_errors['mean']
                    errors_median_cnt[method][i] = cnt_errors['median']
                    errors_95th_cnt[method][i] = cnt_errors['95th']
                    errors_mean_sum[method][i] = sum_errors['mean']
                    errors_median_sum[method][i] = sum_errors['median']
                    errors_95th_sum[method][i] = sum_errors['95th']

                    errors_mean_cnt_del[method][i] = cnt_deleted_errors['mean']
                    errors_median_cnt_del[method][i] = cnt_deleted_errors['median']
                    errors_95th_cnt_del[method][i] = cnt_deleted_errors['95th']
                    errors_mean_sum_del[method][i] = sum_deleted_errors['mean']
                    errors_median_sum_del[method][i] = sum_deleted_errors['median']
                    errors_95th_sum_del[method][i] = sum_deleted_errors['95th']
                    


    """
    #Original
    original = api.run(original_run)
    results = original.summary
    #errors 
    cnt_reals = run.config['ground-truth-count']
    cnt_preds = run.config['predictions-count']

    sum_reals = run.config['ground-truth-sum']
    sum_preds = run.config['predictions-sum']

    cnt_errors = cal_error(cnt_reals, cnt_preds, 'relative-error')
    sum_errors = cal_error(sum_reals, sum_preds, 'relative-error')

    for method in methods:
        errors_mean_cnt[method] = [cnt_errors['mean']] + errors_mean_cnt[method]
        errors_median_cnt[method] = [cnt_errors['median']] + errors_median_cnt[method]
        errors_95th_cnt[method] = [cnt_errors['95th']] + errors_95th_cnt[method]
        errors_mean_sum[method] = [sum_errors['mean']] + errors_mean_sum[method]
        errors_median_sum[method] = [sum_errors['median']] + errors_median_sum[method]
        errors_95th_sum[method] = [sum_errors['95th']] + errors_95th_sum[method]

    """

    # Add data
    palette = plt.get_cmap('Set1')
    markers = ['.', '^', '*', 'x', '+', '4']

    errors_r = [errors_mean_cnt, errors_median_cnt, errors_95th_cnt, errors_mean_sum, errors_median_sum, errors_95th_sum]
    errors_d = [errors_mean_cnt_del, errors_median_cnt_del, errors_95th_cnt_del, errors_mean_sum_del, errors_median_sum_del, errors_95th_sum_del]

    titles_r = ['Count Queries On The Retain Rows', 'Count Queries On The Retain Rows', 'Count Queries On The Retain Rows', 
                'Sum Queries On The Retain Rows', 'Sum Queries On The Retain Rows', 'Sum Queries On The Retain Rows',]
    titles_d = ['Count Queries On The Deleted Rows', 'Count Queries On The Deleted Rows', 'Count Queries On The Deleted Rows', 
                'Sum Queries On The Deleted Rows', 'Sum Queries On The Deleted Rows', 'Sum Queries On The Deleted Rows',]
    y_labels_r = ['Mean Relative Error', 'Median Relative Error', '95th Relative Error']*2 #for count and sum queries
    y_labels_d = ['Mean Absolute Error', 'Median Absolute Error', '95th Absolute Error']*2 #for count and sum queries


    #Delete errors
    labels = ['req1', 'req2', 'req3', 'req4', 'req5']
    for j, error in enumerate(errors_d):
        for i, method in enumerate(methods):
            if method.lower() == 'sgda':
                tag = 'SCRUB'
            elif method.lower() == 'finetune':
                tag = 'Fine-tune'
            else:
                tag = method
            
            plt.plot(labels, error[method], marker=markers[i], color=palette(i), linewidth=2, alpha=1, label=tag)


        plt.legend(loc=2, ncol=2)
         

        plt.title(titles_d[j], loc='center', fontsize=20, fontweight=8, color='black')
        plt.xlabel("delete request", fontsize=18)
        plt.ylabel(y_labels_d[j], fontsize=18)
        plt.xticks(fontsize=16)
        plt.ylim(0, 8)
        # after plotting the data, format the labels
        current_values = plt.gca().get_yticks()
        if (current_values >= 1000).any():
            #plt.gca().yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))


        name = f"results/dbest_{dataset}_{titles_d[j].replace(' ','_')}_{y_labels_d[j].replace(' ', '_')}.png"
        plt.savefig(name)

        plt.clf()

    #Retain errors
    #labels = ['original', 'req1', 'req2', 'req3', 'req4', 'req5']
    labels = ['req1', 'req2', 'req3', 'req4', 'req5']
    for j, error in enumerate(errors_r):
        for i, method in enumerate(methods):
            if method.lower() == 'sgda':
                tag = 'SCRUB'
            elif method.lower() == 'finetune':
                tag = 'Fine-tune'
            else:
                tag = method

            plt.plot(labels, error[method], marker=markers[i], color=palette(i), linewidth=2, alpha=1, label=tag)



        plt.legend(loc=2, ncol=2)
         

        plt.title(titles_r[j], loc='center', fontsize=20, fontweight=4, color='black')
        plt.xlabel("delete request", fontsize=18)
        plt.ylabel(y_labels_r[j], fontsize=18)
        plt.xticks(fontsize=16)
        # after plotting the data, format the labels
        current_values = plt.gca().get_yticks()
        if (current_values >= 1000).any():
            #plt.gca().yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

        name = f"results/dbest_{dataset}_{titles_r[j].replace(' ','_')}_{y_labels_r[j].replace(' ', '_')}.png"
        plt.savefig(name)

        plt.clf()


    #Likelihoods
    labels = ['req1', 'req2', 'req3', 'req4', 'req5']
    for i, method in enumerate(methods):
        if method.lower() == 'sgda':
            tag = 'SCRUB'
        elif method.lower() == 'finetune':
            tag = 'Fine-tune'
        else:
            tag = method

        plt.plot(labels, likelihood[method], marker=markers[i], color=palette(i), linewidth=2, alpha=1, label=tag)



    plt.legend(loc=2, ncol=2)
     

    #plt.title("Likelihood of The Deleted Rows", loc='center', fontsize=14, fontweight=4, color='black')
    plt.xlabel("delete request", fontsize=14)
    plt.ylabel("Likelihood", fontsize=14)
    # after plotting the data, format the labels
    current_values = plt.gca().get_yticks()
    if (current_values >= 1000).any():
        #plt.gca().yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    name = f"results/dbest_{dataset}_likelihood.png"
    plt.savefig(name)

    plt.clf()

def ratio_figures():

    api = wandb.Api()
    entity, project = "PROJECT", "PROJECT"  # set to your entity and project

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

    elif dataset == "dmv":
        runs_oneshot = []

        runs_seq = []

    print ("results for {}".format(dataset))
    metrics_likelihood = ['final-del-likelihood']
    style_names = ['mean-cnt', 'median-cnt', '95th-cnt', 'mean-sum', 'median-sum', '95th-sum']

    errors = {}
    errors_deleted = {}
    for r_j, r_l in zip(runs_oneshot, runs_seq):
        run_j = api.run(r_j)
        run_l = api.run(r_l)
        names_j = run_j.name.split("_")
        names_l = run_l.name.split("_")
        assert names_j[1] == names_l[1]
        method = run_j.config["mode"]

        errs = {}
        errs_deleted = {}
        for case, run in zip(["oneshot", "seq"],[run_j, run_l]):
            results = run.summary
            #errors 
            cnt_reals = run.config['ground-truth-count']
            cnt_preds = run.config['predictions-count']

            sum_reals = run.config['ground-truth-sum']
            sum_preds = run.config['predictions-sum']

            cnt_errors = cal_error(cnt_reals, cnt_preds, 'relative-error')
            sum_errors = cal_error(sum_reals, sum_preds, 'relative-error')


            cnt_deleted_reals = run.config['ground-truth-deleted-count']
            cnt_deleted_preds = run.config['predictions-deleted-count']

            sum_deleted_reals = run.config['ground-truth-deleted-sum']
            sum_deleted_preds = run.config['predictions-deleted-sum']

            cnt_deleted_errors = cal_error(cnt_deleted_reals, cnt_deleted_preds, 'absolute-error')
            sum_deleted_errors = cal_error(sum_deleted_reals, sum_deleted_preds, 'absolute-error')


            errs[case] = {"count": [cnt_errors["mean"], cnt_errors["median"], cnt_errors["95th"]],
                          "sum": [sum_errors["mean"], sum_errors["median"], sum_errors["95th"]]}


            errs_deleted[case] = {"count": [cnt_deleted_errors["mean"], cnt_deleted_errors["median"], cnt_deleted_errors["95th"]], 
                                  "sum": [sum_deleted_errors["mean"], sum_deleted_errors["median"], sum_deleted_errors["95th"]]}

        errors[method] = errs
        errors_deleted[method] = errs_deleted



    for method in errors_deleted.keys():
        row = method + " & "
        for i in range(3):
            row += "{:.2f}".format(errors_deleted[method]["oneshot"]["count"][i]) + " & "
        for i in range(3):
            row += "{:.2f}".format(errors_deleted[method]["oneshot"]["sum"][i]) + " & "
        row = row[:-3]
        print (row + "\\\\")

        row = method + " & "
        for i in range(3):
            row += "{:.2f}".format(errors_deleted[method]["seq"]["count"][i]) + " & "
        for i in range(3):
            row += "{:.2f}".format(errors_deleted[method]["seq"]["sum"][i]) + " & "
        row = row[:-3]
        print (row + "\\\\")



    #Creating ratio figures
    palette = plt.get_cmap('Set1')
    markers = ['.', '^', '*', 'x', '+', '4']

    methods = list(errors.keys())
    for i, method in enumerate(methods):
        if method.lower() == "sgda":
            methods[i] = 'SCRUB'
        elif method.lower() == "retrain":
            methods[i] = 'Retrain'
        elif method.lower() == "finetune":
            methods[i] = 'Fine-tune'
        elif method.lower() == "stale":
            methods[i] = 'Stale'
        elif method.lower() == "neggrad":
            methods[i] = 'NegGrad'
        elif method.lower() == "neggrad+":
            methods[i] = 'NegGrad+'


    methods = tuple(methods)
    labels = ['mean-cnt', 'median-cnt', '99th-cnt', 'mean-sum', 'median-sum', '99th-sum']

    #retain errors
    mean_cnt = [errors[method]["oneshot"]["count"][0]/errors[method]["seq"]["count"][0] for method in errors.keys()]
    med_cnt = [errors[method]["oneshot"]["count"][1]/errors[method]["seq"]["count"][1] for method in errors.keys()]
    th_cnt = [errors[method]["oneshot"]["count"][2]/errors[method]["seq"]["count"][2] for method in errors.keys()]
    mean_sum = [errors[method]["oneshot"]["sum"][0]/errors[method]["seq"]["sum"][0] for method in errors.keys()]
    med_sum = [errors[method]["oneshot"]["sum"][1]/errors[method]["seq"]["sum"][1] for method in errors.keys()]
    th_sum = [errors[method]["oneshot"]["sum"][2]/errors[method]["seq"]["sum"][2] for method in errors.keys()]


    errs_r = [mean_cnt, med_cnt, th_cnt, mean_sum, med_sum, th_sum]
    errors_r = {label:err for label,err in zip(labels,errs_r)}

    #delete errors
    mean_cnt = [errors_deleted[method]["oneshot"]["count"][0]/errors_deleted[method]["seq"]["count"][0] for method in errors_deleted.keys()]
    med_cnt = [errors_deleted[method]["oneshot"]["count"][1]/errors_deleted[method]["seq"]["count"][1] for method in errors_deleted.keys()]
    th_cnt = [errors_deleted[method]["oneshot"]["count"][2]/errors_deleted[method]["seq"]["count"][2] for method in errors_deleted.keys()]
    mean_sum = [errors_deleted[method]["oneshot"]["sum"][0]/errors_deleted[method]["seq"]["sum"][0] for method in errors_deleted.keys()]
    med_sum = [errors_deleted[method]["oneshot"]["sum"][1]/errors_deleted[method]["seq"]["sum"][1] for method in errors_deleted.keys()]
    th_sum = [errors_deleted[method]["oneshot"]["sum"][2]/errors_deleted[method]["seq"]["sum"][2] for method in errors_deleted.keys()]

    errs_d = [mean_cnt, med_cnt, th_cnt, mean_sum, med_sum, th_sum]
    errors_d = {label:err for label,err in zip(labels,errs_d)}


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
    #ax.set_title('the ratio of joint deletion over sequential deletion for different metrics', fontsize=14, fontweight=4, color='black')
    ax.set_xticks(x+0.32)
    ax.set_xticklabels(methods, rotation=45, ha='right', rotation_mode='anchor')
    ax.legend(loc='upper left', ncol=3, fontsize=14)
    #ax.plot(methods, [1]*(len(methods)), "k--")
    ax.axhline(y = 1, color = 'black', linestyle = 'dashed')
    #ax.grid()
    ax.set_ylim(0, 2)

    plt.savefig(f"results/dbest_{dataset}_retained_ratios.png")

    plt.clf()



    x = np.arange(len(methods))  # the label locations
    width = 0.14  # the width of the bars
    multiplier = 0
    palette = plt.get_cmap('Set1')

    fig, ax = plt.subplots(constrained_layout=True)
    ax.tick_params(axis='both', which='major', labelsize=14)


    for i, (metric, errs) in enumerate(errors_d.items()):
        offset = width * multiplier
        rects = ax.bar(x + offset, errs, width, label=metric)
        #ax.bar_label(rects, padding=2)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('one-go/sequential error ratio', fontsize=14, fontweight=4)
    #ax.set_title('the ratio of joint deletion over sequential deletion for different metrics', fontsize=14, fontweight=4, color='black')
    ax.set_xticks(x+0.32)
    ax.set_xticklabels(methods, rotation=45, ha='right', rotation_mode='anchor')
    ax.legend(loc='upper left', ncol=3, fontsize=14)
    #ax.plot(methods, [1]*(len(methods)), "k--")
    ax.axhline(y = 1, color = 'black', linestyle = 'dashed')
    #ax.grid()
    ax.set_ylim(0, 20)

    plt.savefig(f"results/dbest_{dataset}_deleted_ratios.png")

def histogram_plots():
    api = wandb.Api()
    entity, project = "USERNAME", "PROJECT"  # set to your entity and project
    #runs = api.runs(entity+"/"+project)

    dataset = "dmv"
    selective_full = "selective"

    if dataset == 'census':
        if selective_full == "full":
            #census_full
            runs = {'runid': ('model_path', 'filter_path')}

        else:
            #census_selective
            runs = {}


        datafile = '../tabular_data/census/census.csv'
        x_att = 'native_country'
        y_att = 'age'
        x_val_for_hist = ' United-States'

    elif dataset == 'forest':
        if selective_full == "full":
            #forest_full
            runs = {}


        else:
            #forest_selective
            runs = {}




        datafile = '../tabular_data/forest/forest.csv'
        x_att = 'slope'
        y_att = 'elevation'
        x_val_for_hist = '11'

    elif dataset == 'dmv':
        if selective_full == "full":
            #dmv_full
            runs = {}


        else:
            #dmv_selective
            runs = {}



        datafile = '../tabular_data/DMV/dmv_for_mdn.csv'
        x_att = 'body_type'
        y_att = 'max_gross_weight'
        x_val_for_hist = 'PICK'



    for r in runs.keys():
        
        #wandb won't download the file if there are already files with the same name
        assert not os.path.exists(runs[r][0])
        if runs[r][1] is not None:
            assert not os.path.exists(runs[r][1])

        model = wandb.restore(runs[r][0], run_path=r)
        if runs[r][1] is not None:
            filters = wandb.restore(runs[r][1], run_path=r)
        

        run = api.run(r)
        mode = run.config['mode']
        names = run.name.replace('-','_').split("_")

        if mode.lower() == "train":
            benchmarking.plot_histogram(model.name, datafile, x_att, y_att, x_val_for_hist, dataset+"_"+mode, mode, None)
        else:
            data_handler = Dataset.Data(datafile, x_attributes=[x_att], y_attributes=[y_att], sep=',')
            _, _ = data_handler.delete_data(filters.name)
            benchmarking.plot_histogram(model.name, datafile.replace('.csv','_reduced.csv'), x_att, y_att, x_val_for_hist, dataset+"_"+mode+f"_{selective_full}", mode, None)



        os.remove(runs[r][0])
        if runs[r][1] is not None:
            os.remove(runs[r][1])

if __name__ == "__main__":
    #tables()
    #seq_figures()
    ratio_figures()
    #histogram_plots()

