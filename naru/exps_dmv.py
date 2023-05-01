import os
from pathlib import Path

dataset = "dmv"
datafile = "../tabular_data/DMV/DMV.csv"
model_pth = "models/dmv-train-20.4MB-model25.112-data19.489-made-resmade-hidden256_256_256_256_256-emb32-directIo-binaryInone_hotOut-inputNoEmbIfLeq-colmask-20epochs-seed0.pt"

min_max_vals = [(7000, 7250), (7000, 7500), (7000, 7700), (7000, 8000), (7000, 8200)]

def eval_intermediates():

    retrain_pre_model = model_pth
    stale_pre_model = model_pth
    ft_pre_model = model_pth
    ngplus_pre_model = model_pth
    ng_pre_model = model_pth
    scrub_pre_model = model_pth


    filters_temp = """{
        "filters": [
                {
                    "type":"range",
                    "att": "Maximum Gross Weight",
                    "min_val": MINVALUEHERE,
                    "max_val": MAXVALUEHERE
                }
        ]
    }
    """

    wandb_config = " --wandb_mode=online --wandb_project=naru-unlearning --wandb_entity=USERNAME --wandb_group_name={}"

    retrain_command = "python train_model.py --mode=retrain --filters={} --num-gpus=1 --dataset=dmv --epochs=20 --warmups=200 --bs=256 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.0002 "
    stale_command = "python train_model.py --mode=stale --filters={} --num-gpus=1 --dataset=dmv --epochs=0 --warmups=200 --bs=256 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.0002 --pre-model={} "
    ft_command1 = "python train_model.py --mode=finetune --filters={} --num-gpus=1 --dataset=dmv --epochs=10 --warmups=0 --bs=256 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.00002 --pre-model={} "
    ft_command2 = "python train_model.py --mode=finetune --filters={} --num-gpus=1 --dataset=dmv --epochs=10 --warmups=0 --bs=256 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate={} --pre-model={} "
    ng_command1 = "python train_model.py --mode=negatgrad --filters={} --num-gpus=1 --dataset=dmv --epochs=5 --warmups=0 --bs=256 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.00002  --alpha=0.9999 --del-bs=128 --pre-model={} "
    ng_command2 = "python train_model.py --mode=negatgrad+ --filters={} --num-gpus=1 --dataset=dmv --epochs=2 --warmups=0 --bs=256 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.000002 --alpha=0.0 --del-bs=128 --pre-model={} "
    scrub_command1 = "python train_model.py --mode=scrubs --filters={} --num-gpus=1 --dataset=dmv --epochs=8 --warmups=0 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.00002 --alpha=0.9 --del-bs=256 --bs=128 --msteps=3 --pre-model={} "

    os.system("rm previous_queries.pkl")
    os.system("cp ../tabular_data/DMV/DMV_stable.csv ../tabular_data/DMV/DMV.csv")
    for i, min_max in enumerate(min_max_vals):
        filters = filters_temp.replace("MINVALUEHERE", str(min_max[0]))
        filters = filters.replace("MAXVALUEHERE", str(min_max[1]))
        filters_path = f"filters_{dataset}_temp.json"
        with open(filters_path, 'w') as f:
            f.write(filters)

        if i > 0:
            #reduced_data = datafile.replace(".csv", "_reduced.csv")
            #os.rename(reduced_data, datafile)

            #getting the latest models from the previous unlearn operations
            paths = sorted(Path("models/").iterdir(), key=os.path.getmtime, reverse=True)
            for p in paths:
                if str(p).split('/')[1].split('-')[0].lower() == dataset.lower():
                    if str(p).split('/')[1].split('-')[1].lower() == 'retrain':
                        retrain_pre_model = str(p)
                        break
            for p in paths:
                if str(p).split('/')[1].split('-')[0].lower() == dataset.lower():
                    if str(p).split('/')[1].split('-')[1].lower() == 'stale':
                        stale_pre_model = str(p)
                        break
            for p in paths:
                if str(p).split('/')[1].split('-')[0].lower() == dataset.lower():
                    if str(p).split('/')[1].split('-')[1].lower() == 'finetune':
                        print(p)
                        ft_pre_model = str(p)
                        break
            for p in paths:
                if str(p).split('/')[1].split('-')[0].lower() == dataset.lower():
                    if str(p).split('/')[1].split('-')[1].lower() == 'negatgrad':
                        ng_pre_model = str(p)
                        break
            for p in paths:
                if str(p).split('/')[1].split('-')[0].lower() == dataset.lower():
                    if str(p).split('/')[1].split('-')[1].lower() == 'scrubs':
                        scrub_pre_model = str(p)
                        break



        wandb_command = wandb_config.format(f"dmv-req{i+1}")
        
        retrain_command = retrain_command.format(filters_path, retrain_pre_model) + wandb_command
        stale_command = stale_command.format(filters_path, stale_pre_model) + wandb_command
        ft_lr = 0.00002*(10^(i+1))
        ft_command1 = ft_command1.format(filters_path, ft_pre_model) + wandb_command
        ng_command1 = ng_command1.format(filters_path, ng_pre_model) + wandb_command
        ng_command2 = ng_command2.format(filters_path, ng_pre_model) + wandb_command
        scrub_command1 = scrub_command1.format(filters_path, scrub_pre_model) + wandb_command
        

        #os.system(retrain_command)
        #os.system(stale_command)
        #os.system(ft_command1)
        #os.system(ng_command1)
        os.system(ng_command2)
        #os.system(scrub_command1)

        os.system("rm previous_queries.pkl")
    os.system("cp ../tabular_data/DMV/DMV_stable.csv ../tabular_data/DMV/DMV.csv")

def eval_lastpoints():

    def del_all():
        retrain_pre_model = model_pth
        stale_pre_model = model_pth
        ft_pre_model = model_pth
        ngplus_pre_model = model_pth
        ng_pre_model = model_pth
        scrub_pre_model = model_pth


        filters_temp = """{
            "filters": [
                    {
                        "type":"range_full",
                        "att": "Maximum Gross Weight",
                        "min_val": MINVALUEHERE,
                        "max_val": MAXVALUEHERE
                    }
            ]
        }
        """

        wandb_config = " --wandb_mode={} --wandb_project=naru-unlearning-largerbench --wandb_entity=USERNAME --wandb_group_name=dmv-compare-last-points-v1"

        retrain_command = "python train_model.py --mode=retrain --filters={} --num-gpus=1 --dataset=dmv --epochs=20 --warmups=200 --bs=256 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.0002 "
        stale_command = "python train_model.py --mode=stale --filters={} --num-gpus=1 --dataset=dmv --epochs=0 --warmups=200 --bs=256 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.0002 --pre-model={} "
        ft_command1 = "python train_model.py --mode=finetune --filters={} --num-gpus=1 --dataset=dmv --epochs=10 --warmups=0 --bs=256 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.00002 --pre-model={} "
        ft_command2 = "python train_model.py --mode=finetune --filters={} --num-gpus=1 --dataset=dmv --epochs=10 --warmups=0 --bs=256 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.000002 --pre-model={} "
        ng_command1 = "python train_model.py --mode=negatgrad --filters={} --num-gpus=1 --dataset=dmv --epochs=5 --warmups=0 --bs=256 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.00002  --alpha=0.9999 --del-bs=128 --pre-model={} "
        ng_command2 = "python train_model.py --mode=negatgrad+ --filters={} --num-gpus=1 --dataset=dmv --epochs=2 --warmups=0 --bs=256 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.000002 --alpha=0.0 --del-bs=128 --pre-model={} "
        scrub_command1 = "python train_model.py --mode=scrub --filters={} --num-gpus=1 --dataset=dmv --epochs=8 --warmups=0 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.00002 --alpha=0.9 --del-bs=256 --bs=128 --msteps=3 --pre-model={} "


        os.system("rm previous_queries.pkl")
        os.system("cp ../tabular_data/DMV/DMV_stable.csv ../tabular_data/DMV/DMV.csv")

        filters = filters_temp.replace("MINVALUEHERE", str(min_max_vals[-1][0]))
        filters = filters.replace("MAXVALUEHERE", str(min_max_vals[-1][1]))
        filters_path = f"filters_{dataset}_temp.json"
        with open(filters_path, 'w') as f:
            f.write(filters)


        wandb_command = wandb_config.format("online")

        
        retrain_command = retrain_command.format(filters_path, retrain_pre_model) + wandb_command
        stale_command = stale_command.format(filters_path, stale_pre_model) + wandb_command
        ft_command1 = ft_command1.format(filters_path, ft_pre_model) + wandb_command
        ft_command2 = ft_command2.format(filters_path, ft_pre_model) + wandb_command
        ng_command1 = ng_command1.format(filters_path, ng_pre_model) + wandb_command
        ng_command2 = ng_command2.format(filters_path, ng_pre_model) + wandb_command
        scrub_command1 = scrub_command1.format(filters_path, scrub_pre_model) + wandb_command
        

        os.system(retrain_command)
        os.system(stale_command)
        os.system(ft_command1)
        os.system(ft_command2)
        os.system(ng_command1)
        os.system(ng_command2)
        os.system(scrub_command1)


        os.system("cp ../tabular_data/DMV/DMV_stable.csv ../tabular_data/DMV/DMV.csv")


    def del_seq():

        retrain_pre_model = model_pth
        stale_pre_model = model_pth
        ft_pre_model = model_pth
        ngplus_pre_model = model_pth
        ng_pre_model = model_pth
        scrub_pre_model = model_pth


        filters_temp = """{
            "filters": [
                    {
                        "type":"range_full",
                        "att": "Maximum Gross Weight",
                        "min_val": MINVALUEHERE,
                        "max_val": MAXVALUEHERE
                    }
            ]
        }
        """

        wandb_config = " --wandb_mode={} --wandb_project=naru-unlearning-largerbench --wandb_entity=USERNAME --wandb_group_name=dmv-compare-last-points-v1"

        retrain_command = "python train_model.py --mode=retrain --filters={} --num-gpus=1 --dataset=dmv --epochs=20 --warmups=200 --bs=256 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.0002 "
        stale_command = "python train_model.py --mode=stale --filters={} --num-gpus=1 --dataset=dmv --epochs=0 --warmups=200 --bs=256 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.0002 --pre-model={} "
        ft_command1 = "python train_model.py --mode=finetune --filters={} --num-gpus=1 --dataset=dmv --epochs=10 --warmups=0 --bs=256 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.00002 --pre-model={} "
        ft_command2 = "python train_model.py --mode=finetune --filters={} --num-gpus=1 --dataset=dmv --epochs=10 --warmups=0 --bs=256 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate={} --pre-model={} "
        ng_command1 = "python train_model.py --mode=negatgrad --filters={} --num-gpus=1 --dataset=dmv --epochs=5 --warmups=0 --bs=256 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.00002  --alpha=0.9999 --del-bs=128 --pre-model={} "
        ng_command2 = "python train_model.py --mode=negatgrad+ --filters={} --num-gpus=1 --dataset=dmv --epochs=2 --warmups=0 --bs=256 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.000002 --alpha=0.0 --del-bs=128 --pre-model={} "
        scrub_command1 = "python train_model.py --mode=scrub --filters={} --num-gpus=1 --dataset=dmv --epochs=8 --warmups=0 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.00002 --alpha=0.9 --del-bs=256 --bs=128 --msteps=3 --pre-model={} "


        os.system("cp ../tabular_data/DMV/DMV_stable.csv ../tabular_data/DMV/DMV.csv")
        for i, min_max in enumerate(min_max_vals):
            filters = filters_temp.replace("MINVALUEHERE", str(min_max[0]))
            filters = filters.replace("MAXVALUEHERE", str(min_max[1]))
            filters_path = f"filters_{dataset}_temp.json"
            with open(filters_path, 'w') as f:
                f.write(filters)

            if i > 0:
                #reduced_data = datafile.replace(".csv", "_reduced.csv")
                #os.rename(reduced_data, datafile)

                #getting the latest models from the previous unlearn operations
                paths = sorted(Path("models/").iterdir(), key=os.path.getmtime, reverse=True)
                for p in paths:
                    if str(p).split('/')[1].split('-')[0].lower() == dataset.lower():
                        if str(p).split('/')[1].split('-')[1].lower() == 'retrain':
                            retrain_pre_model = str(p)
                            break
                for p in paths:
                    if str(p).split('/')[1].split('-')[0].lower() == dataset.lower():
                        if str(p).split('/')[1].split('-')[1].lower() == 'stale':
                            stale_pre_model = str(p)
                            break
                for p in paths:
                    if str(p).split('/')[1].split('-')[0].lower() == dataset.lower():
                        if str(p).split('/')[1].split('-')[1].lower() == 'finetune':
                            print(p)
                            ft_pre_model = str(p)
                            break
                for p in paths:
                    if str(p).split('/')[1].split('-')[0].lower() == dataset.lower():
                        if str(p).split('/')[1].split('-')[1].lower() == 'negatgrad':
                            ng_pre_model = str(p)
                            break
                for p in paths:
                    if str(p).split('/')[1].split('-')[0].lower() == dataset.lower():
                        if str(p).split('/')[1].split('-')[1].lower() == 'scrubs':
                            scrub_pre_model = str(p)
                            break


            if i == len(min_max_vals)-1:
                wandb_command = wandb_config.format("online")
            else:
                wandb_command = wandb_config.format("disabled")

            
            retrain_command = retrain_command.format(filters_path, retrain_pre_model) + wandb_command
            stale_command = stale_command.format(filters_path, stale_pre_model) + wandb_command
            ft_command1 = ft_command1.format(filters_path, ft_pre_model) + wandb_command
            ft_lr = 0.00002*(10^(i+1))
            ft_command2 = ft_command2.format(filters_path, ft_lr, ft_pre_model) + wandb_command
            ng_command1 = ng_command1.format(filters_path, ng_pre_model) + wandb_command
            ng_command2 = ng_command2.format(filters_path, ng_pre_model) + wandb_command
            scrub_command1 = scrub_command1.format(filters_path, scrub_pre_model) + wandb_command
            

            os.system(retrain_command)
            os.system(stale_command)
            os.system(ft_command1)
            os.system(ft_command2)
            os.system(ng_command1)
            os.system(ng_command2)
            os.system(scrub_command1)


        os.system("cp ../tabular_data/DMV/DMV_stable.csv ../tabular_data/DMV/DMV.csv")

    del_all()
    del_seq()


def run_regular():

    retrain_pre_model = model_pth
    stale_pre_model = model_pth
    ft_pre_model = model_pth
    ngplus_pre_model = model_pth
    ng_pre_model = model_pth
    scrub_pre_model = model_pth


    filters_temp = """{
        "filters": [
                {
                    "type":"range_selective",
                    "att": "Maximum Gross Weight",
                    "min_val": MINVALUEHERE,
                    "max_val": MAXVALUEHERE
                }
        ]
    }
    """

    wandb_config = " --wandb_mode={} --wandb_project=naru-unlearning-largerbench --wandb_entity=USERNAME --wandb_group_name=dmv-v1"
    train_command = "python train_model.py --mode=train --num-gpus=1 --dataset=dmv --epochs=20 --warmups=8000 --bs=2048 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.0002 "
    retrain_command = "python train_model.py --mode=retrain --filters={} --num-gpus=1 --dataset=dmv --epochs=20 --warmups=8000 --bs=2048 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.0002 "
    stale_command = "python train_model.py --mode=stale --filters={} --num-gpus=1 --dataset=dmv --epochs=0 --warmups=200 --bs=256 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.0002 --pre-model={} "
    ft_command1 = "python train_model.py --mode=finetune --filters={} --num-gpus=1 --dataset=dmv --epochs=5 --warmups=0 --bs=2048 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.000002 --pre-model={} "    
    ng_command1 = "python train_model.py --mode=negatgrad+ --filters={} --num-gpus=1 --dataset=dmv --epochs=5 --warmups=0 --bs=2048 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.000002 --alpha=0.9 --del-bs=1024 --pre-model={} "
    ng_command2 = "python train_model.py --mode=negatgrad --filters={} --num-gpus=1 --dataset=dmv --epochs=5 --warmups=0 --bs=2048 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.000002 --alpha=0.0 --del-bs=1024 --pre-model={} "
    scrub_command1 = "python train_model.py --mode=scrub --filters={} --num-gpus=1 --dataset=dmv --epochs=5 --warmups=0 --bs=2048 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.00002 --alpha=0.9 --del-bs=2048 --msteps=2 --pre-model={} "


    os.system("rm previous_queries.pkl")
    os.system("cp ../tabular_data/DMV/DMV_stable.csv ../tabular_data/DMV/DMV.csv")

    filters = filters_temp.replace("MINVALUEHERE", str(min_max_vals[-1][0]))
    filters = filters.replace("MAXVALUEHERE", str(min_max_vals[-1][1]))
    filters_path = f"filters_{dataset}_temp.json"
    with open(filters_path, 'w') as f:
        f.write(filters)


    wandb_command = wandb_config.format("online")

    train_command = train_command + wandb_command
    retrain_command = retrain_command.format(filters_path, retrain_pre_model) + wandb_command
    stale_command = stale_command.format(filters_path, stale_pre_model) + wandb_command
    ft_command1 = ft_command1.format(filters_path, ft_pre_model) + wandb_command
    ng_command1 = ng_command1.format(filters_path, ng_pre_model) + wandb_command
    ng_command2 = ng_command2.format(filters_path, ng_pre_model) + wandb_command
    scrub_command1 = scrub_command1.format(filters_path, scrub_pre_model) + wandb_command
    

    #os.system(train_command)
    os.system(retrain_command)
    os.system(stale_command)
    os.system(ft_command1)
    os.system(ng_command1)
    os.system(ng_command2)
    os.system(scrub_command1)


    os.system("cp ../tabular_data/DMV/DMV_stable.csv ../tabular_data/DMV/DMV.csv")

if __name__ == "__main__":
    #run_regular()
    eval_lastpoints()
    #eval_intermediates()
    


