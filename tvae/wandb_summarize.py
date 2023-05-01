import wandb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import scipy.stats as st
from pathlib import Path
import os

from ctgan import data as data_handler
import benchmarking

def confidence_intervals():
    api = wandb.Api()
    entity, project = "USERNAME", "PROJECT"  # set to your entity and project
    runs = api.runs(entity+"/"+project)

    dataset = "dmv"
    delete_filter = "equality"
    delete_frac = 0.5


    for run in runs:
        if run.config['dataset'] == dataset and run.group == f'{dataset}-new':
            try:
                run.config['delete-filter']
            except:
                run.config['delete-filter'] = "train"

            if (run.config['delete-filter'] == delete_filter and run.config['delete_frac'] == delete_frac) or run.config['mode'] == 'Train':

                results = run.summary

                f1_real = [i*100 for i in results['f1 real']]
                f1_synth = [i*100 for i in results['f1 synth']]

                if run.config['mode'] != 'Train':
                    f1_real_delete = [i*100 for i in results['deleted f1 real']]
                    f1_synth_delete = [i*100 for i in results['deleted f1 synth']]                   


                interval_real = st.t.interval(alpha=0.95, df=len(f1_real)-1, loc=np.mean(f1_real), scale=st.sem(f1_real))
                interval_synth = st.t.interval(alpha=0.95, df=len(f1_synth)-1, loc=np.mean(f1_synth), scale=st.sem(f1_synth))
                if run.config['mode'] != 'Train':
                    interval_real_deleted = st.t.interval(alpha=0.95, df=len(f1_real_delete)-1, loc=np.mean(f1_real_delete), scale=st.sem(f1_real_delete))
                    interval_synth_deleted = st.t.interval(alpha=0.95, df=len(f1_synth_delete)-1, loc=np.mean(f1_synth_delete), scale=st.sem(f1_synth_delete))

                if run.config['mode'] != 'Train':
                    row = run.config['mode'] + '&'
                    row = row + "{:.2f}".format(np.mean(interval_real)) + "±" + "{:.2f}".format(interval_real[1]-np.mean(interval_real)) + '&'
                    row = row + "{:.2f}".format(np.mean(interval_synth)) + "±" + "{:.2f}".format(interval_synth[1]-np.mean(interval_synth)) + '&'
                    row = row + "{:.2f}".format(np.mean(interval_real_deleted)) + "±" + "{:.2f}".format(interval_real_deleted[1]-np.mean(interval_real_deleted)) + '&'
                    row = row + "{:.2f}".format(np.mean(interval_synth_deleted)) + "±" + "{:.2f}".format(interval_synth_deleted[1]-np.mean(interval_synth_deleted))
                    row = row + "\\\\"
                    print (row)
                else:
                    row = run.config['mode'] + '&'
                    row = row + "{:.2f}".format(np.mean(interval_real)) + "±" + "{:.2f}".format(interval_real[1]-np.mean(interval_real)) + '&'
                    row = row + "{:.2f}".format(np.mean(interval_synth)) + "±" + "{:.2f}".format(interval_synth[1]-np.mean(interval_synth))
                    row = row + "\\\\"
                    print (row)



if __name__ == "__main__":
    confidence_intervals()                 
