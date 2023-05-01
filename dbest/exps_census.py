import os

dataset = "census"
datafile = "../tabular_data/census/census.csv"
model_pth = "models/census_train_native_country_age_seed1234.dill"

min_max_vals = [(30, 31), (32, 33), (34, 35), (36, 37), (38, 39)]


def eval_intermediates():
    retrain_pre_model = model_pth
    stale_pre_model = model_pth
    ft_pre_model = model_pth
    ngplus_pre_model = model_pth
    ng_pre_model = model_pth
    SCRUB_pre_model = model_pth


    filters_temp = """{
        "filters": [
                {
                    "type":"range_full",
                    "att": "age",
                    "min_val": MINVALUEHERE,
                    "max_val": MAXVALUEHERE
                }
        ]
    }
    """

    wandb_config = " --wandb_mode={} --wandb_project=mdn-unlearning-largerbench --wandb_entity=USERNAME --wandb_group_name={}"
    eval_command = " --evaluate --eval_per_epoch=999 --num_eval_queries=500 --compare_hist --x_val_for_hist=' United-States'"
    eval_del_command = " --eval_deleted --eval_likelihood"
    use_pre_queries = " --use_pre_queries"

    train_command = "python train_mdn.py --dataset=census --mode=train --x_att=native_country --y_att=age --datafile={} --learning_rate=0.001 --lr_decay_epochs=25,35,40 --epochs=50 --batch_size=128"
    retrain_command = "python train_mdn.py --dataset=census --mode=retrain --x_att=native_country --y_att=age --datafile={} --filters='{}' --learning_rate=0.001 --lr_decay_epochs=25,35,40 --epochs=50 --batch_size=128 --pre_model={} --tr_frac=1"

    stale_command = "python train_mdn.py --dataset=census --mode=stale --x_att=native_country --y_att=age --datafile={} --filters={} --learning_rate=0.001 --lr_decay_epochs=25,35,40 --epochs=50 --batch_size=128 --pre_model={} --tr_frac=1 "
    ft_command = "python train_mdn.py --dataset=census --mode=finetune --x_att=native_country --y_att=age --datafile={} --filters={} --batch_size=128 --pre_model={} --tr_frac=1 --learning_rate=0.001 --lr_decay_epochs=4,6,8 --epochs=10 "
    ngplus_command = "python train_mdn.py --dataset=census --mode=NegGrad+ --x_att=native_country --y_att=age --datafile={} --filters={} --batch_size=128 --pre_model={} --tr_frac=1 --learning_rate=0.001 --lr_decay_epochs=4,6,8 --epochs=10 --alpha=0.99 "
    ng_command = "python train_mdn.py --dataset=census --mode=NegGrad --x_att=native_country --y_att=age --datafile={} --filters={} --batch_size=128 --pre_model={} --tr_frac=1 --learning_rate=0.0001 --lr_decay_epochs=4,6,8 --epochs=10 --alpha=0.0 "
    SCRUB_command = "python train_mdn.py --dataset=census --mode=SCRUB --x_att=native_country --y_att=age --datafile={} --filters={} --batch_size=64 --del_batch_size=128 --pre_model={} --tr_frac=1 --learning_rate=0.0001 --lr_decay_epochs=4,6,8 --epochs=10 --msteps=3 --alpha=0.99 --beta=0 "


    os.system("cp ../tabular_data/census/census_stable.csv ../tabular_data/census/census.csv")
    for i, min_max in enumerate(min_max_vals):
        filters = filters_temp.replace("MINVALUEHERE", str(min_max[0]))
        filters = filters.replace("MAXVALUEHERE", str(min_max[1]))
        filters_path = f"filters_{dataset}_temp.json"
        with open(filters_path, 'w') as f:
            f.write(filters)

        if i > 0:
            reduced_data = datafile.replace(".csv", "_reduced.csv")
            os.rename(reduced_data, datafile)

            retrain_pre_model = model_pth.replace("_train_", "_retrain_")
            stale_pre_model = model_pth.replace("_train_", "_stale_")
            ft_pre_model = model_pth.replace("_train_", "_finetune_")
            ngplus_pre_model = model_pth.replace("_train_", "_NegGrad+_")
            ng_pre_model = model_pth.replace("_train_", "_NegGrad_")
            SCRUB_pre_model = model_pth.replace("_train_", "_SCRUB_")

        wandb_command = wandb_config.format("online", f"census-req{i+1}")

        train_command = train_command.format(datafile) + eval_command + wandb_command
        retrain_command = retrain_command.format(datafile, filters_path, retrain_pre_model) + eval_command + eval_del_command + wandb_command
        stale_command = stale_command.format(datafile, filters_path, stale_pre_model) + eval_command + eval_del_command + use_pre_queries + wandb_command
        ft_command = ft_command.format(datafile, filters_path, ft_pre_model) + eval_command + eval_del_command + use_pre_queries + wandb_command
        ngplus_command = ngplus_command.format(datafile, filters_path, ngplus_pre_model) + eval_command + eval_del_command + use_pre_queries + wandb_command
        ng_command = ng_command.format(datafile, filters_path, ng_pre_model) + eval_command + eval_del_command + use_pre_queries + wandb_command
        SCRUB_command = SCRUB_command.format(datafile, filters_path, SCRUB_pre_model) + eval_command + eval_del_command + use_pre_queries + wandb_command
        


        if i==0:
            os.system(train_command)

        print (filters_path)
        os.system(retrain_command)
        os.system(stale_command)
        os.system(ft_command)
        os.system(ngplus_command)
        os.system(ng_command)
        os.system(SCRUB_command)

    os.system("cp ../tabular_data/census/census_stable.csv ../tabular_data/census/census.csv")

def eval_lastpoints():
    retrain_pre_model = model_pth
    stale_pre_model = model_pth
    ft_pre_model = model_pth
    ngplus_pre_model = model_pth
    ng_pre_model = model_pth
    SCRUB_pre_model = model_pth


    filters_temp = """{
        "filters": [
                {
                    "type":"range_full",
                    "att": "age",
                    "min_val": MINVALUEHERE,
                    "max_val": MAXVALUEHERE
                }
        ]
    }
    """

    wandb_config = " --wandb_mode={} --wandb_project=mdn-unlearning-largerbench --wandb_entity=USERNAME --wandb_group_name=census-oneshot-vs-seq"
    eval_command = " --evaluate --eval_per_epoch=999 --num_eval_queries=2000 --compare_hist --x_val_for_hist=' United-States'"
    eval_del_command = " --eval_deleted --eval_likelihood"
    use_pre_queries = " --use_pre_queries"

    train_command = "python train_mdn.py --dataset=census --mode=train --x_att=native_country --y_att=age --datafile={} --learning_rate=0.001 --lr_decay_epochs=25,35,40 --epochs=50 --batch_size=128"
    retrain_command = "python train_mdn.py --dataset=census --mode=retrain --x_att=native_country --y_att=age --datafile={} --filters='{}' --learning_rate=0.001 --lr_decay_epochs=25,35,40 --epochs=50 --batch_size=128 --pre_model={} --tr_frac=1"

    stale_command = "python train_mdn.py --dataset=census --mode=stale --x_att=native_country --y_att=age --datafile={} --filters={} --learning_rate=0.001 --lr_decay_epochs=25,35,40 --epochs=50 --batch_size=128 --pre_model={} --tr_frac=1 "
    ft_command = "python train_mdn.py --dataset=census --mode=finetune --x_att=native_country --y_att=age --datafile={} --filters={} --batch_size=128 --pre_model={} --tr_frac=1 --learning_rate=0.001 --lr_decay_epochs=4,6,8 --epochs=10 "
    ngplus_command = "python train_mdn.py --dataset=census --mode=NegGrad+ --x_att=native_country --y_att=age --datafile={} --filters={} --batch_size=128 --pre_model={} --tr_frac=1 --learning_rate=0.001 --lr_decay_epochs=4,6,8 --epochs=10 --alpha=0.99 "
    ng_command = "python train_mdn.py --dataset=census --mode=NegGrad --x_att=native_country --y_att=age --datafile={} --filters={} --batch_size=128 --pre_model={} --tr_frac=1 --learning_rate=0.0001 --lr_decay_epochs=4,6,8 --epochs=10 --alpha=0.0 "
    SCRUB_command = "python train_mdn.py --dataset=census --mode=SCRUB --x_att=native_country --y_att=age --datafile={} --filters={} --batch_size=64 --del_batch_size=128 --pre_model={} --tr_frac=1 --learning_rate=0.0001 --lr_decay_epochs=4,6,8 --epochs=10 --msteps=3 --alpha=0.99 --beta=0 "


    def del_one_go(filters_temp, train_command, retrain_command, stale_command, ft_command, ngplus_command, ng_command, SCRUB_command, wandb_config):
        os.system("cp ../tabular_data/census/census_stable.csv ../tabular_data/census/census.csv")
        filters = filters_temp.replace("MINVALUEHERE", str(min_max_vals[0][0]))
        filters = filters.replace("MAXVALUEHERE", str(min_max_vals[-1][1]))
        filters_path = f"filters_{dataset}_temp.json"
        with open(filters_path, 'w') as f:
            f.write(filters)

        wandb_command = wandb_config.format("online")

        train_command = train_command.format(datafile) + eval_command + wandb_command
        retrain_command = retrain_command.format(datafile, filters_path, retrain_pre_model) + eval_command + eval_del_command + wandb_command
        stale_command = stale_command.format(datafile, filters_path, stale_pre_model) + eval_command + eval_del_command + use_pre_queries + wandb_command
        ft_command = ft_command.format(datafile, filters_path, ft_pre_model) + eval_command + eval_del_command + use_pre_queries + wandb_command
        ngplus_command = ngplus_command.format(datafile, filters_path, ngplus_pre_model) + eval_command + eval_del_command + use_pre_queries + wandb_command
        ng_command = ng_command.format(datafile, filters_path, ng_pre_model) + eval_command + eval_del_command + use_pre_queries + wandb_command
        SCRUB_command = SCRUB_command.format(datafile, filters_path, SCRUB_pre_model) + eval_command + eval_del_command + use_pre_queries + wandb_command
        

        #os.system(train_command)
        os.system(retrain_command)
        os.system(stale_command)
        os.system(ft_command)
        os.system(ngplus_command)
        os.system(ng_command)
        os.system(SCRUB_command) 


    def del_seq(filters_temp, train_command, retrain_command, stale_command, ft_command, ngplus_command, ng_command, SCRUB_command, wandb_config, 
                    retrain_pre_model, stale_pre_model, ft_pre_model, ngplus_pre_model, ng_pre_model, SCRUB_pre_model):
        os.system("cp ../tabular_data/census/census_stable.csv ../tabular_data/census/census.csv")
        for i, min_max in enumerate(min_max_vals):
            filters = filters_temp.replace("MINVALUEHERE", str(min_max[0]))
            filters = filters.replace("MAXVALUEHERE", str(min_max[1]))
            filters_path = f"filters_{dataset}_temp.json"
            with open(filters_path, 'w') as f:
                f.write(filters)

            if i > 0:
                reduced_data = datafile.replace(".csv", "_reduced.csv")
                os.rename(reduced_data, datafile)

                retrain_pre_model = model_pth.replace("_train_", "_retrain_")
                stale_pre_model = model_pth.replace("_train_", "_stale_")
                ft_pre_model = model_pth.replace("_train_", "_finetune_")
                ngplus_pre_model = model_pth.replace("_train_", "_NegGrad+_")
                ng_pre_model = model_pth.replace("_train_", "_NegGrad_")
                SCRUB_pre_model = model_pth.replace("_train_", "_SCRUB_")

            if i == len(min_max_vals)-1:
                wandb_command = wandb_config.format("online")

                train_command = train_command.format(datafile) + eval_command + wandb_command
                retrain_command = retrain_command.format(datafile, filters_path, retrain_pre_model) + eval_command + eval_del_command + use_pre_queries + wandb_command
                stale_command = stale_command.format(datafile, filters_path, stale_pre_model) + eval_command + eval_del_command + use_pre_queries + wandb_command
                ft_command = ft_command.format(datafile, filters_path, ft_pre_model) + eval_command + eval_del_command + use_pre_queries + wandb_command
                ngplus_command = ngplus_command.format(datafile, filters_path, ngplus_pre_model) + eval_command + eval_del_command + use_pre_queries + wandb_command
                ng_command = ng_command.format(datafile, filters_path, ng_pre_model) + eval_command + eval_del_command + use_pre_queries + wandb_command
                SCRUB_command = SCRUB_command.format(datafile, filters_path, SCRUB_pre_model) + eval_command + eval_del_command + use_pre_queries + wandb_command

            else:
                wandb_command = wandb_config.format("disabled")

                train_command = train_command.format(datafile) + wandb_command
                retrain_command = retrain_command.format(datafile, filters_path, retrain_pre_model) + wandb_command
                stale_command = stale_command.format(datafile, filters_path, stale_pre_model) + wandb_command
                ft_command = ft_command.format(datafile, filters_path, ft_pre_model) + wandb_command
                ngplus_command = ngplus_command.format(datafile, filters_path, ngplus_pre_model) + wandb_command
                ng_command = ng_command.format(datafile, filters_path, ng_pre_model) + wandb_command
                SCRUB_command = SCRUB_command.format(datafile, filters_path, SCRUB_pre_model) + wandb_command
            

            os.system(retrain_command)
            os.system(stale_command)
            os.system(ft_command)
            os.system(ngplus_command)
            os.system(ng_command)
            os.system(SCRUB_command)


    #first delete all together and create benchmark to evaluate
    del_one_go(filters_temp, train_command, retrain_command, stale_command, ft_command, ngplus_command, ng_command, SCRUB_command, wandb_config)
    #Then delete incrementally and evaluate the last point
    del_seq(filters_temp, train_command, retrain_command, stale_command, ft_command, ngplus_command, ng_command, SCRUB_command, wandb_config,
                retrain_pre_model, stale_pre_model, ft_pre_model, ngplus_pre_model, ng_pre_model, SCRUB_pre_model)

    os.system("cp ../tabular_data/census/census_stable.csv ../tabular_data/census/census.csv")

def run_regular():
    retrain_pre_model = model_pth
    stale_pre_model = model_pth
    ft_pre_model = model_pth
    ngplus_pre_model = model_pth
    ng_pre_model = model_pth
    SCRUB_pre_model = model_pth


    filters_temp = """{
        "filters": [
                {
                    "type":"range_selective",
                    "att": "age",
                    "min_val": MINVALUEHERE,
                    "max_val": MAXVALUEHERE
                }
        ]
    }
    """

    wandb_config = " --wandb_mode={} --wandb_project=mdn-unlearning-largerbench --wandb_entity=USERNAME --wandb_group_name=census-v1"
    eval_command = " --evaluate --eval_per_epoch=999 --num_eval_queries=2000 --compare_hist --x_val_for_hist=' United-States'"
    eval_del_command = " --eval_likelihood"#" --eval_deleted --eval_likelihood" #if filter type is range_selective, only use " --eval_likelihood"
    use_pre_queries = " --use_pre_queries"

    train_command = "python train_mdn.py --dataset=census --mode=train --x_att=native_country --y_att=age --datafile={} --learning_rate=0.001 --lr_decay_epochs=25,35,40 --epochs=50 --batch_size=128"
    retrain_command = "python train_mdn.py --dataset=census --mode=retrain --x_att=native_country --y_att=age --datafile={} --filters='{}' --learning_rate=0.001 --lr_decay_epochs=25,35,40 --epochs=50 --batch_size=128 --pre_model={} --tr_frac=1"

    stale_command = "python train_mdn.py --dataset=census --mode=stale --x_att=native_country --y_att=age --datafile={} --filters={} --learning_rate=0.001 --lr_decay_epochs=25,35,40 --epochs=50 --batch_size=128 --pre_model={} --tr_frac=1  "
    ft_command = "python train_mdn.py --dataset=census --mode=finetune --x_att=native_country --y_att=age --datafile={} --filters={} --batch_size=128 --pre_model={} --tr_frac=1 --learning_rate=0.001 --lr_decay_epochs=4,6,8 --epochs=10  "
    ngplus_command = "python train_mdn.py --dataset=census --mode=NegGrad+ --x_att=native_country --y_att=age --datafile={} --filters={} --batch_size=128 --pre_model={} --tr_frac=1 --learning_rate=0.001 --lr_decay_epochs=4,6,8 --epochs=10 --alpha=0.99 "
    ng_command = "python train_mdn.py --dataset=census --mode=NegGrad --x_att=native_country --y_att=age --datafile={} --filters={} --batch_size=128 --pre_model={} --tr_frac=1 --learning_rate=0.0001 --lr_decay_epochs=4,6,8 --epochs=10 --alpha=0.0  "
    SCRUB_command = "python train_mdn.py --dataset=census --mode=SCRUB --x_att=native_country --y_att=age --datafile={} --filters={} --batch_size=64 --del_batch_size=128 --pre_model={} --tr_frac=1 --learning_rate=0.0001 --lr_decay_epochs=4,6,8 --epochs=10 --msteps=5 --alpha=0.99 --beta=0 "



    os.system("cp ../tabular_data/census/census_stable.csv ../tabular_data/census/census.csv")
    filters = filters_temp.replace("MINVALUEHERE", str(min_max_vals[0][0]))
    filters = filters.replace("MAXVALUEHERE", str(min_max_vals[-1][1]))
    filters_path = f"filters_{dataset}_temp.json"
    with open(filters_path, 'w') as f:
        f.write(filters)

    wandb_command = wandb_config.format("online")

    train_command = train_command.format(datafile) + eval_command + wandb_command
    retrain_command = retrain_command.format(datafile, filters_path, retrain_pre_model) + eval_command + eval_del_command + wandb_command
    stale_command = stale_command.format(datafile, filters_path, stale_pre_model) + eval_command + eval_del_command + use_pre_queries + wandb_command
    ft_command = ft_command.format(datafile, filters_path, ft_pre_model) + eval_command + eval_del_command + use_pre_queries + wandb_command
    ngplus_command = ngplus_command.format(datafile, filters_path, ngplus_pre_model) + eval_command + eval_del_command + use_pre_queries + wandb_command
    ng_command = ng_command.format(datafile, filters_path, ng_pre_model) + eval_command + eval_del_command + use_pre_queries + wandb_command
    SCRUB_command = SCRUB_command.format(datafile, filters_path, SCRUB_pre_model) + eval_command + eval_del_command + use_pre_queries + wandb_command
    

    os.system(train_command)
    os.system(retrain_command)
    os.system(stale_command)
    os.system(ft_command)
    os.system(ngplus_command)
    os.system(ng_command)
    os.system(SCRUB_command) 

    os.system("cp ../tabular_data/census/census_stable.csv ../tabular_data/census/census.csv")
    
if __name__ == "__main__":
    #run_regular()
    #eval_lastpoints()
    eval_intermediates()
    
    







