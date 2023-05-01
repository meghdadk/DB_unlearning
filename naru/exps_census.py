import os
from pathlib import Path

dataset = "census"
datafile = "../tabular_data/census/census.csv"
model_pth = "models/census-train-38.5MB-model30.489-data15.573-made-resmade-hidden256_256_256_256_256-emb32-directIo-binaryInone_hotOut-inputNoEmbIfLeq-colmask-20epochs-seed0.pt"

min_max_vals = [(30, 31), (30, 32), (30, 33), (30, 34), (30, 35)]

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
                    "type":"range_full",
                    "att": "age",
                    "min_val": MINVALUEHERE,
                    "max_val": MAXVALUEHERE
                }
        ]
    }
    """

    wandb_config = " --wandb_mode={} --wandb_project=naru-unlearning-largerbench --wandb_entity=USERNAME --wandb_group_name={}"

    retrain_command = "python train_model.py --mode=retrain --filters={} --num-gpus=1 --dataset=census --epochs=20 --warmups=200 --bs=128 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.001 "
    stale_command = "python train_model.py --mode=stale --filters={} --num-gpus=1 --dataset=census --epochs=0 --warmups=200 --bs=128 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.001 --pre-model={} "
    ft_command1 = "python train_model.py --mode=finetune --filters={} --num-gpus=1 --dataset=census --epochs=10 --warmups=0 --bs=128 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate={} --pre-model={} "
    ft_command2 = "python train_model.py --mode=finetune --filters={} --num-gpus=1 --dataset=census --epochs=10 --warmups=0 --bs=128 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate={} --pre-model={} "
    ng_command1 = "python train_model.py --mode=negatgrad --filters={} --num-gpus=1 --dataset=census --epochs=10 --warmups=0 --bs=128 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.0001 --alpha=0.95 --pre-model={} "
    ng_command2 = "python train_model.py --mode=negatgrad --filters={} --num-gpus=1 --dataset=census --epochs=5 --warmups=0 --bs=128 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.0001 --alpha=0.85 --del-bs=64 --pre-model={} "
    ng_command3 = "python train_model.py --mode=negatgrad --filters={} --num-gpus=1 --dataset=census --epochs=5 --warmups=0 --bs=128 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.0001 --alpha=0.8 --del-bs=32 --pre-model={} "
    ng_command4 = "python train_model.py --mode=negatgrad --filters={} --num-gpus=1 --dataset=census --epochs=5 --warmups=0 --bs=128 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.00001 --alpha=0 --del-bs=128 --pre-model={} "
    scrub_command1 = "python train_model.py --mode=scrub --filters={} --num-gpus=1 --dataset=census --epochs=8 --warmups=0 --bs=128 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.0001 --alpha=0.9 --bs=64 --del-bs=64 --msteps=3 --pre-model={} "
    scrub_command2 = "python train_model.py --mode=scrub --filters={} --num-gpus=1 --dataset=census --epochs=8 --warmups=0 --bs=128 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.0001 --alpha=0.5 --bs=64 --del-bs=64 --msteps=3 --pre-model={} "


    os.system("rm previous_queries.pkl")
    os.system("cp ../tabular_data/census/census_stable.csv ../tabular_data/census/census.csv")
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


        wandb_command = wandb_config.format("online", f"census-req{i+1}")

        
        retrain_command = retrain_command.format(filters_path, retrain_pre_model) + wandb_command
        stale_command = stale_command.format(filters_path, stale_pre_model) + wandb_command
        ft_lr = 0.000001*(10^(i+1))
        ft_command1 = ft_command1.format(filters_path, ft_lr, ft_pre_model) + wandb_command
        ft_command2 = ft_command2.format(filters_path, ft_lr, ft_pre_model) + wandb_command
        ng_command1 = ng_command1.format(filters_path, ng_pre_model) + wandb_command
        ng_command2 = ng_command2.format(filters_path, ng_pre_model) + wandb_command
        ng_command3 = ng_command3.format(filters_path, ng_pre_model) + wandb_command
        ng_command4 = ng_command4.format(filters_path, ng_pre_model) + wandb_command
        scrub_command1 = scrub_command1.format(filters_path, scrub_pre_model) + wandb_command
        scrub_command2 = scrub_command2.format(filters_path, scrub_pre_model) + wandb_command
        

        os.system(retrain_command)
        os.system(stale_command)
        os.system(ft_command1)
        os.system(ft_command2)
        os.system(ng_command1)
        os.system(ng_command2)
        os.system(ng_command3)
        os.system(ng_command4)
        os.system(scrub_command1)
        os.system(scrub_command2)

        os.system("rm previous_queries.pkl")
    os.system("cp ../tabular_data/census/census_stable.csv ../tabular_data/census/census.csv")

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
                        "att": "age",
                        "min_val": MINVALUEHERE,
                        "max_val": MAXVALUEHERE
                    }
            ]
        }
        """

        wandb_config = " --wandb_mode={} --wandb_project=naru-unlearning-largerbench --wandb_entity=USERNAME --wandb_group_name={}"

        retrain_command = "python train_model.py --mode=retrain --filters={} --num-gpus=1 --dataset=census --epochs=20 --warmups=200 --bs=128 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.001 "
        stale_command = "python train_model.py --mode=stale --filters={} --num-gpus=1 --dataset=census --epochs=0 --warmups=200 --bs=128 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.001 --pre-model={} "
        ft_command2 = "python train_model.py --mode=finetune --filters={} --num-gpus=1 --dataset=census --epochs=10 --warmups=0 --bs=128 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.00001 --pre-model={} "
        ng_command1 = "python train_model.py --mode=negatgrad --filters={} --num-gpus=1 --dataset=census --epochs=10 --warmups=0 --bs=128 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.0001 --alpha=0.95 --pre-model={} "
        ng_command4 = "python train_model.py --mode=negatgrad --filters={} --num-gpus=1 --dataset=census --epochs=5 --warmups=0 --bs=128 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.00001 --alpha=0 --del-bs=128 --pre-model={} "
        scrub_command2 = "python train_model.py --mode=scrub --filters={} --num-gpus=1 --dataset=census --epochs=8 --warmups=0 --bs=128 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.0001 --alpha=0.5 --bs=64 --del-bs=64 --msteps=3 --pre-model={} "


        os.system("rm previous_queries.pkl")
        os.system("cp ../tabular_data/census/census_stable.csv ../tabular_data/census/census.csv")
        for i, min_max in enumerate([min_max_vals[-1]]):
            filters = filters_temp.replace("MINVALUEHERE", str(min_max[0]))
            filters = filters.replace("MAXVALUEHERE", str(min_max[1]))
            filters_path = f"filters_{dataset}_temp.json"
            with open(filters_path, 'w') as f:
                f.write(filters)


            wandb_command = wandb_config.format("online", f"census-compare-last-points")

            
            retrain_command = retrain_command.format(filters_path, retrain_pre_model) + wandb_command
            stale_command = stale_command.format(filters_path, stale_pre_model) + wandb_command
            ft_command2 = ft_command2.format(filters_path, ft_pre_model) + wandb_command
            ng_command1 = ng_command1.format(filters_path, ng_pre_model) + wandb_command
            ng_command4 = ng_command4.format(filters_path, ng_pre_model) + wandb_command
            scrub_command2 = scrub_command2.format(filters_path, scrub_pre_model) + wandb_command
            

            os.system(retrain_command)
            os.system(stale_command)
            os.system(ft_command2)
            os.system(ng_command1)
            os.system(ng_command4)
            os.system(scrub_command2)


        os.system("cp ../tabular_data/census/census_stable.csv ../tabular_data/census/census.csv")


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
                        "att": "age",
                        "min_val": MINVALUEHERE,
                        "max_val": MAXVALUEHERE
                    }
            ]
        }
        """

        wandb_config = " --wandb_mode={} --wandb_project=naru-unlearning-largerbench --wandb_entity=USERNAME --wandb_group_name={}"

        retrain_command = "python train_model.py --mode=retrain --filters={} --num-gpus=1 --dataset=census --epochs=20 --warmups=200 --bs=128 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.001 "
        stale_command = "python train_model.py --mode=stale --filters={} --num-gpus=1 --dataset=census --epochs=0 --warmups=200 --bs=128 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.001 --pre-model={} "
        ft_command2 = "python train_model.py --mode=finetune --filters={} --num-gpus=1 --dataset=census --epochs=10 --warmups=0 --bs=128 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate={} --pre-model={} "
        ng_command1 = "python train_model.py --mode=negatgrad --filters={} --num-gpus=1 --dataset=census --epochs=10 --warmups=0 --bs=128 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.0001 --alpha=0.95 --pre-model={} "
        ng_command4 = "python train_model.py --mode=negatgrad --filters={} --num-gpus=1 --dataset=census --epochs=5 --warmups=0 --bs=128 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.00001 --alpha=0 --del-bs=128 --pre-model={} "
        scrub_command1 = "python train_model.py --mode=scrub --filters={} --num-gpus=1 --dataset=census --epochs=8 --warmups=0 --bs=128 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.0001 --alpha=0.9 --bs=64 --del-bs=64 --msteps=3 --pre-model={} "
        scrub_command2 = "python train_model.py --mode=scrub --filters={} --num-gpus=1 --dataset=census --epochs=8 --warmups=0 --bs=128 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.0001 --alpha=0.5 --bs=64 --del-bs=64 --msteps=3 --pre-model={} "


        os.system("cp ../tabular_data/census/census_stable.csv ../tabular_data/census/census.csv")
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
                wandb_command = wandb_config.format("online", f"census-compare-last-points")
            else:
                wandb_command = wandb_config.format("disabled", f"census-compare-last-points")

            
            retrain_command = retrain_command.format(filters_path, retrain_pre_model) + wandb_command
            stale_command = stale_command.format(filters_path, stale_pre_model) + wandb_command
            ft_lr = 0.000001*(10^(i+1))
            ft_command2 = ft_command2.format(filters_path, ft_lr, ft_pre_model) + wandb_command
            ng_command1 = ng_command1.format(filters_path, ng_pre_model) + wandb_command
            ng_command4 = ng_command4.format(filters_path, ng_pre_model) + wandb_command
            scrub_command2 = scrub_command2.format(filters_path, scrub_pre_model) + wandb_command
            

            os.system(retrain_command)
            os.system(stale_command)
            os.system(ft_command2)
            os.system(ng_command1)
            os.system(ng_command4)
            os.system(scrub_command2)


        os.system("cp ../tabular_data/census/census_stable.csv ../tabular_data/census/census.csv")

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
                    "att": "age",
                    "min_val": MINVALUEHERE,
                    "max_val": MAXVALUEHERE
                }
        ]
    }
    """

    wandb_config = " --wandb_mode={} --wandb_project=naru-unlearning-largerbench --wandb_entity=USERNAME --wandb_group_name=census-v1"

    train_command = "python train_model.py --mode=train --num-gpus=1 --dataset=census --epochs=20 --warmups=200 --bs=128 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.001 "
    retrain_command = "python train_model.py --mode=retrain --filters={} --num-gpus=1 --dataset=census --epochs=20 --warmups=200 --bs=128 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.001 "
    stale_command = "python train_model.py --mode=stale --filters={} --num-gpus=1 --dataset=census --epochs=0 --warmups=200 --bs=128 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.001 --pre-model={} "
    ft_command2 = "python train_model.py --mode=finetune --filters={} --num-gpus=1 --dataset=census --epochs=10 --warmups=0 --bs=128 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.00001 --pre-model={} "
    ng_command1 = "python train_model.py --mode=negatgrad --filters={} --num-gpus=1 --dataset=census --epochs=10 --warmups=0 --bs=128 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.0001 --alpha=0.95 --pre-model={} "
    ng_command4 = "python train_model.py --mode=negatgrad --filters={} --num-gpus=1 --dataset=census --epochs=5 --warmups=0 --bs=128 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.00001 --alpha=0 --del-bs=128 --pre-model={} "
    scrub_command2 = "python train_model.py --mode=scrub --filters={} --num-gpus=1 --dataset=census --epochs=8 --warmups=0 --bs=128 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking --eval-per-epoch=999 --learning-rate=0.0001 --alpha=0.5 --bs=64 --del-bs=64 --msteps=3 --pre-model={} "


    os.system("rm previous_queries.pkl")
    os.system("cp ../tabular_data/census/census_stable.csv ../tabular_data/census/census.csv")

    filters = filters_temp.replace("MINVALUEHERE", str(min_max_vals[-1][0]))
    filters = filters.replace("MAXVALUEHERE", str(min_max_vals[-1][1]))
    filters_path = f"filters_{dataset}_temp.json"
    with open(filters_path, 'w') as f:
        f.write(filters)


    wandb_command = wandb_config.format("online")

    train_command = train_command + wandb_command
    retrain_command = retrain_command.format(filters_path, retrain_pre_model) + wandb_command
    stale_command = stale_command.format(filters_path, stale_pre_model) + wandb_command
    ft_command2 = ft_command2.format(filters_path, ft_pre_model) + wandb_command
    ng_command1 = ng_command1.format(filters_path, ng_pre_model) + wandb_command
    ng_command4 = ng_command4.format(filters_path, ng_pre_model) + wandb_command
    scrub_command2 = scrub_command2.format(filters_path, scrub_pre_model) + wandb_command
    

    #os.system(train_command)
    #os.system(retrain_command)
    os.system(stale_command)
    os.system(ft_command2)
    os.system(ng_command1)
    os.system(ng_command4)
    os.system(scrub_command2)


    os.system("cp ../tabular_data/census/census_stable.csv ../tabular_data/census/census.csv")

if __name__ == "__main__":
    #run_regular()
    eval_intermediates()
    #eval_lastpoints()

