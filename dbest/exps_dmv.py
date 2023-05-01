import os

dataset = "dmv"
datafile = "../tabular_data/DMV/dmv_for_mdn.csv"
model_pth = "models/dmv_train_body_type_max_gross_weight_seed1234.dill"

min_max_vals = [(7000, 7300), (7301, 7600), (7601, 7800), (7801, 8000), (8001, 8200)]

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
                    "type":"range",
                    "att": "max_gross_weight",
                    "min_val": MINVALUEHERE,
                    "max_val": MAXVALUEHERE
                }
        ]
    }
    """


    wandb_config = " --wandb_mode={} --wandb_project=mdn-unlearning --wandb_entity=USERNAME --wandb_group_name={}"
    eval_command = " --evaluate --eval_per_epoch=999 --num_eval_queries=500 --compare_hist --x_val_for_hist=PICK"
    eval_del_command = " --eval_deleted --eval_likelihood"
    use_pre_queries = " --use_pre_queries"

    train_command = "python train_mdn.py --mode=train --dataset=dmv --datafile={} --x_att=body_type --y_att=max_gross_weight --mode=train --learning_rate=0.001 --lr_decay_epochs=10,20,30 --epochs=50 --batch_size=128 --num_hid_layers=2 --hid_layer_sizes=128,128 --num_gaussians=80 "
    retrain_command = "python train_mdn.py --mode=retrain --dataset=dmv --datafile={} --filters={} --x_att=body_type --y_att=max_gross_weight --learning_rate=0.001 --lr_decay_epochs=25,35,40 --epochs=50 --batch_size=128 --num_hid_layers=2 --hid_layer_sizes=128,128 --num_gaussians=80 --pre_model={}"

    stale_command = "python train_mdn.py --mode=stale --dataset=dmv --datafile={} --filters={} --x_att=body_type --y_att=max_gross_weight  --pre_model={} "
    ft_command = "python train_mdn.py --mode=finetune --dataset=dmv --datafile={} --filters={} --x_att=body_type --y_att=max_gross_weight --learning_rate=0.001 --lr_decay_epochs=4,6,8 --epochs=10 --batch_size=128 --pre_model={} "
    ngplus1_command = "python train_mdn.py --mode=NegGrad+ --dataset=dmv --datafile={} --filters={} --x_att=body_type --y_att=max_gross_weight --learning_rate=0.001 --lr_decay_epochs=4,6,8 --epochs=10 --batch_size=128 --alpha=0.95 --pre_model={} "
    ngplus2_command = "python train_mdn.py --mode=NegGrad+ --dataset=dmv --datafile={} --filters={} --x_att=body_type --y_att=max_gross_weight --learning_rate=0.001 --lr_decay_epochs=4,6,8 --epochs=10 --batch_size=128 --alpha=0.98 --pre_model={} "
    ng_command = "python train_mdn.py --mode=NegGrad --dataset=dmv --datafile={} --filters={} --x_att=body_type --y_att=max_gross_weight --learning_rate=0.00001 --lr_decay_epochs=4,6,8 --epochs=10 --batch_size=128 --alpha=0 --pre_model={} "
    SCRUB_command = "python train_mdn.py --mode=SCRUB --dataset=dmv --datafile={} --filters={} --x_att=body_type --y_att=max_gross_weight --learning_rate=0.0001 --lr_decay_epochs=4,6,8 --epochs=10 --pre_model={} --alpha=0.9 --beta=0 --batch_size=64 --del_batch_size=128 "


    os.system("cp ../tabular_data/DMV/dmv_for_mdn_stable.csv ../tabular_data/DMV/dmv_for_mdn.csv")
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

        wandb_command = wandb_config.format("online", f"dmv-req{i+1}")

        train_command = train_command.format(datafile) + eval_command + wandb_command
        retrain_command = retrain_command.format(datafile, filters_path, retrain_pre_model) + eval_command + eval_del_command + wandb_command
        stale_command = stale_command.format(datafile, filters_path, stale_pre_model) + eval_command + eval_del_command + use_pre_queries + wandb_command
        ft_command = ft_command.format(datafile, filters_path, ft_pre_model) + eval_command + eval_del_command + use_pre_queries + wandb_command
        ngplus1_command = ngplus1_command.format(datafile, filters_path, ngplus_pre_model) + eval_command + eval_del_command + use_pre_queries + wandb_command
        ngplus2_command = ngplus2_command.format(datafile, filters_path, ngplus_pre_model) + eval_command + eval_del_command + use_pre_queries + wandb_command
        ng_command = ng_command.format(datafile, filters_path, ng_pre_model) + eval_command + eval_del_command + use_pre_queries + wandb_command
        SCRUB_command = SCRUB_command.format(datafile, filters_path, SCRUB_pre_model) + eval_command + eval_del_command + use_pre_queries + wandb_command
        



        if i==0:
            pass#os.system(train_command)

        print (filters_path)
        os.system(retrain_command)
        os.system(stale_command)
        os.system(ft_command)
        os.system(ngplus1_command)
        os.system(ngplus2_command)
        os.system(ng_command)
        os.system(SCRUB_command)


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
                    "att": "max_gross_weight",
                    "min_val": MINVALUEHERE,
                    "max_val": MAXVALUEHERE
                }
        ]
    }
    """

    wandb_config = " --wandb_mode={} --wandb_project=mdn-unlearning-largerbench --wandb_entity=USERNAME --wandb_group_name=dmv-oneshot-vs-seq"
    eval_command = " --evaluate --eval_per_epoch=999 --num_eval_queries=2000 --compare_hist --x_val_for_hist=PICK"
    eval_del_command = " --eval_deleted --eval_likelihood"
    use_pre_queries = " --use_pre_queries"

    train_command = "python train_mdn.py --mode=train --dataset=dmv --datafile={} --x_att=body_type --y_att=max_gross_weight --mode=train --learning_rate=0.001 --lr_decay_epochs=10,20,30 --epochs=50 --batch_size=128 --num_hid_layers=2 --hid_layer_sizes=128,128 --num_gaussians=80 "
    retrain_command = "python train_mdn.py --mode=retrain --dataset=dmv --datafile={} --filters={} --x_att=body_type --y_att=max_gross_weight --learning_rate=0.001 --lr_decay_epochs=25,35,40 --epochs=50 --batch_size=128 --num_hid_layers=2 --hid_layer_sizes=128,128 --num_gaussians=80 --pre_model={}"

    stale_command = "python train_mdn.py --mode=stale --dataset=dmv --datafile={} --filters={} --x_att=body_type --y_att=max_gross_weight --pre_model={} "
    ft_command = "python train_mdn.py --mode=finetune --dataset=dmv --datafile={} --filters={} --x_att=body_type --y_att=max_gross_weight --learning_rate=0.001 --lr_decay_epochs=4,6,8 --epochs=10 --batch_size=128 --pre_model={} "
    ngplus1_command = "python train_mdn.py --mode=NegGrad+ --dataset=dmv --datafile={} --filters={} --x_att=body_type --y_att=max_gross_weight --learning_rate=0.001 --lr_decay_epochs=4,6,8 --epochs=10 --batch_size=128 --alpha=0.95 --pre_model={} "
    ngplus2_command = "python train_mdn.py --mode=NegGrad+ --dataset=dmv --datafile={} --filters={} --x_att=body_type --y_att=max_gross_weight --learning_rate=0.001 --lr_decay_epochs=4,6,8 --epochs=10 --batch_size=128 --alpha=0.98 --pre_model={} "
    ng_command = "python train_mdn.py --mode=NegGrad --dataset=dmv --datafile={} --filters={} --x_att=body_type --y_att=max_gross_weight --learning_rate=0.001 --lr_decay_epochs=4,6,8 --epochs=10 --batch_size=128 --alpha=0 --pre_model={} "
    SCRUB_command = "python train_mdn.py --mode=SCRUB --dataset=dmv --datafile={} --filters={} --x_att=body_type --y_att=max_gross_weight --learning_rate=0.0001 --lr_decay_epochs=4,6,8 --epochs=10 --msteps=2 --pre_model={} --alpha=0.9999 --beta=0 --batch_size=64 --del_batch_size=128 "


    def dell_all(filters_temp, train_command, retrain_command, stale_command, ft_command, ngplus1_command, ngplus2_command, ng_command, SCRUB_command, wandb_config):
        os.system("cp ../tabular_data/DMV/dmv_for_mdn_stable.csv ../tabular_data/DMV/dmv_for_mdn.csv")
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
        ngplus1_command = ngplus1_command.format(datafile, filters_path, ngplus_pre_model) + eval_command + eval_del_command + use_pre_queries + wandb_command
        ngplus2_command = ngplus2_command.format(datafile, filters_path, ngplus_pre_model) + eval_command + eval_del_command + use_pre_queries + wandb_command
        ng_command = ng_command.format(datafile, filters_path, ng_pre_model) + eval_command + eval_del_command + use_pre_queries + wandb_command
        SCRUB_command = SCRUB_command.format(datafile, filters_path, SCRUB_pre_model) + eval_command + eval_del_command + use_pre_queries + wandb_command
        

        #os.system(train_command)
        os.system(retrain_command)
        os.system(stale_command)
        os.system(ft_command)
        os.system(ngplus1_command)
        os.system(ngplus2_command)
        os.system(ng_command)
        os.system(SCRUB_command) 


    def del_seq(filters_temp, train_command, retrain_command, stale_command, ft_command, ngplus1_command, ngplus2_command, ng_command, SCRUB_command, wandb_config, 
                    retrain_pre_model, stale_pre_model, ft_pre_model, ngplus_pre_model, ng_pre_model, SCRUB_pre_model):
        os.system("cp ../tabular_data/DMV/dmv_for_mdn_stable.csv ../tabular_data/DMV/dmv_for_mdn.csv")
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
                ngplus1_command = ngplus1_command.format(datafile, filters_path, ngplus_pre_model) + eval_command + eval_del_command + use_pre_queries + wandb_command
                ngplus2_command = ngplus2_command.format(datafile, filters_path, ngplus_pre_model) + eval_command + eval_del_command + use_pre_queries + wandb_command
                ng_command = ng_command.format(datafile, filters_path, ng_pre_model) + eval_command + eval_del_command + use_pre_queries + wandb_command
                SCRUB_command = SCRUB_command.format(datafile, filters_path, SCRUB_pre_model) + eval_command + eval_del_command + use_pre_queries + wandb_command

            else:
                wandb_command = wandb_config.format("disabled")

                train_command = train_command.format(datafile) + wandb_command
                retrain_command = retrain_command.format(datafile, filters_path, retrain_pre_model) + wandb_command
                stale_command = stale_command.format(datafile, filters_path, stale_pre_model) + wandb_command
                ft_command = ft_command.format(datafile, filters_path, ft_pre_model) + wandb_command
                ngplus1_command = ngplus1_command.format(datafile, filters_path, ngplus_pre_model) + wandb_command
                ngplus2_command = ngplus2_command.format(datafile, filters_path, ngplus_pre_model) + wandb_command
                ng_command = ng_command.format(datafile, filters_path, ng_pre_model) + wandb_command
                SCRUB_command = SCRUB_command.format(datafile, filters_path, SCRUB_pre_model) + wandb_command                
            

            os.system(retrain_command)
            os.system(stale_command)
            os.system(ft_command)
            os.system(ngplus1_command)
            os.system(ngplus2_command)
            os.system(ng_command)
            os.system(SCRUB_command)


    #first delete all together and create benchmark to evaluate
    dell_all(filters_temp, train_command, retrain_command, stale_command, ft_command, ngplus1_command, ngplus2_command, ng_command, SCRUB_command, wandb_config)
    #Then delete incrementally and evaluate the last point
    del_seq(filters_temp, train_command, retrain_command, stale_command, ft_command, ngplus1_command, ngplus2_command, ng_command, SCRUB_command, wandb_config,
                retrain_pre_model, stale_pre_model, ft_pre_model, ngplus_pre_model, ng_pre_model, SCRUB_pre_model)


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
                    "att": "max_gross_weight",
                    "min_val": MINVALUEHERE,
                    "max_val": MAXVALUEHERE
                }
        ]
    }
    """

    wandb_config = " --wandb_mode={} --wandb_project=mdn-unlearning-largerbench --wandb_entity=USERNAME --wandb_group_name=dmv-v1"
    eval_command = " --evaluate --eval_per_epoch=999 --num_eval_queries=2000 --compare_hist --x_val_for_hist=PICK"
    eval_del_command = "  --eval_likelihood"#" --eval_deleted --eval_likelihood"
    use_pre_queries = " --use_pre_queries"

    train_command = "python train_mdn.py --mode=train --dataset=dmv --datafile={} --x_att=body_type --y_att=max_gross_weight --learning_rate=0.001 --lr_decay_epochs=10,20,25 --epochs=30 --batch_size=128 --num_hid_layers=2 --hid_layer_sizes=128,128 --num_gaussians=30 "
    retrain_command = "python train_mdn.py --mode=retrain --dataset=dmv --datafile={} --filters={} --x_att=body_type --y_att=max_gross_weight --learning_rate=0.001 --lr_decay_epochs=10,20,25 --epochs=30 --batch_size=128 --num_hid_layers=2 --hid_layer_sizes=128,128 --num_gaussians=30 --pre_model={}"

    stale_command = "python train_mdn.py --mode=stale --dataset=dmv --datafile={} --filters={} --x_att=body_type --y_att=max_gross_weight --pre_model={} "
    ft_command = "python train_mdn.py --mode=finetune --dataset=dmv --datafile={} --filters={} --x_att=body_type --y_att=max_gross_weight --learning_rate=0.001 --lr_decay_epochs=4,6,8 --epochs=10 --batch_size=128 --pre_model={} "
    ngplus1_command = "python train_mdn.py --mode=NegGrad+ --dataset=dmv --datafile={} --filters={} --x_att=body_type --y_att=max_gross_weight --learning_rate=0.001 --lr_decay_epochs=4,6,8 --epochs=10 --batch_size=128 --alpha=0.95 --pre_model={} "
    ngplus2_command = "python train_mdn.py --mode=NegGrad+ --dataset=dmv --datafile={} --filters={} --x_att=body_type --y_att=max_gross_weight --learning_rate=0.001 --lr_decay_epochs=4,6,8 --epochs=10 --batch_size=128 --alpha=0.98 --pre_model={} "
    ng_command = "python train_mdn.py --mode=NegGrad --dataset=dmv --datafile={} --filters={} --x_att=body_type --y_att=max_gross_weight --learning_rate=0.001 --lr_decay_epochs=4,6,8 --epochs=10 --batch_size=128 --alpha=0 --pre_model={} "
    SCRUB_command = "python train_mdn.py --mode=SCRUB --dataset=dmv --datafile={} --filters={} --x_att=body_type --y_att=max_gross_weight --learning_rate=0.0001 --lr_decay_epochs=4,6,8 --epochs=10 --msteps=2 --pre_model={} --alpha=0.9999 --beta=0 --batch_size=64 --del_batch_size=128 "


    os.system("cp ../tabular_data/DMV/dmv_for_mdn_stable.csv ../tabular_data/DMV/dmv_for_mdn.csv")
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
    ngplus1_command = ngplus1_command.format(datafile, filters_path, ngplus_pre_model) + eval_command + eval_del_command + use_pre_queries + wandb_command
    ngplus2_command = ngplus2_command.format(datafile, filters_path, ngplus_pre_model) + eval_command + eval_del_command + use_pre_queries + wandb_command
    ng_command = ng_command.format(datafile, filters_path, ng_pre_model) + eval_command + eval_del_command + use_pre_queries + wandb_command
    SCRUB_command = SCRUB_command.format(datafile, filters_path, SCRUB_pre_model) + eval_command + eval_del_command + use_pre_queries + wandb_command
    

    os.system(train_command)
    os.system(retrain_command)
    os.system(stale_command)
    os.system(ft_command)
    os.system(ngplus1_command)
    os.system(ngplus2_command)
    os.system(ng_command)
    os.system(SCRUB_command) 



if __name__ == "__main__":
    #run_regular()
    eval_lastpoints()
    #eval_intermediates()









