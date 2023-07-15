import copy
import os
import time
import numpy as np
from itertools import cycle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import models
import datasets
from data import DataProcessor
import utils
from KD import DistillKL


DATASET = 'census'
MODEL = 'resnet'
DATA_PATH = '../tabular_data/census/census.csv'
DELETE_FILTER = 'census_filters_selective.json'
COLS = [
    'age','workclass','fnlwgt','education',
    'marital_status','occupation','relationship',
    'race','sex','capital_gain','capital_loss',
    'hours_per_week','native_country'
]
CAT_COLS = ['workclass', 'education', 'occupation', 'relationship', 'race', 'sex', 'native_country']
NUM_COLS = ['age', 'fnlwgt', 'capital_gain', 'capital_loss', 'hours_per_week']
LABEL = 'marital_status'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#SEED = 8

for SEED in [8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]:

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    def train(model_init, 
            train_loader, 
            val_loader,  
            checkpoint_path,
            device='cpu', 
            seed=1):
        
        print ("="*30,"> training the original/retrain model ...")
        model = copy.deepcopy(model_init)
        # Define the loss function with class weighting
        learning_rate = 0.01
        num_epochs = 50
        #criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        checkpoint_path += f'epochs{num_epochs}-lr{learning_rate}-seed{seed}.pth'
        training_time = 0
        for epoch in range(num_epochs):
            model.train()
            t1 = time.time()
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()

            scheduler.step()
            t2 = time.time()
            training_time += t2 - t1

            val_accuracy = utils.accuracy(model, val_loader, device)
            train_accuracy = utils.accuracy(model, train_loader, device)

            print(f'Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}')

        state = model.state_dict()
        torch.save(state, checkpoint_path)
        print('Saved a checkpoint at {}'.format(checkpoint_path))
        
        return model, training_time

    def fine_tune(original, 
                retain_loader, 
                val_loader, 
                checkpoint_path, 
                device='cpu',
                seed=1):
        print ("="*30,"> unlearning by fine-tune ...")
        model_ft = copy.deepcopy(original)
        # Define the loss function with class weighting
        learning_rate = 0.01
        num_epochs = 10
        #criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model_ft.parameters(), lr=learning_rate)
        #scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        checkpoint_path += f'epochs{num_epochs}-lr{learning_rate}-seed{seed}.pth'   
        training_time = 0
        for epoch in range(num_epochs):
            model_ft.train()
            t1 = time.time()
            for inputs, targets in retain_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()

                outputs = model_ft(inputs)
                loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()

            # Update the learning rate
            #scheduler.step()

            t2 = time.time()
            training_time += t2 - t1
            val_accuracy = utils.accuracy(model_ft, val_loader, device)
            train_accuracy = utils.accuracy(model_ft, retain_loader, device)

            print(f'Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        state = model_ft.state_dict()
        torch.save(state, checkpoint_path)
        print('Saved a checkpoint at {}'.format(checkpoint_path))

        return model_ft, training_time

    def neggrad(original, 
                forget_loader, 
                val_loader, 
                checkpoint_path, 
                device='cpu',
                seed=1):
        print ("="*30,"> unlearning by NegGrad ...")
        model_ng = copy.deepcopy(original)
        # Define the loss function with class weighting
        learning_rate = 0.01
        num_epochs = 10
        #criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model_ng.parameters(), lr=learning_rate)
        #scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        checkpoint_path += f'epochs{num_epochs}-lr{learning_rate}-seed{seed}.pth'
        training_time = 0
        for epoch in range(num_epochs):
            model_ng.train()
            t1 = time.time()
            for inputs, targets in forget_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()


                outputs = model_ng(inputs)
                loss = -1 * criterion(outputs, targets)


                loss.backward()
                optimizer.step()

            # Update the learning rate
            #scheduler.step()

            t2 = time.time()
            training_time += t2 - t1

            val_accuracy = utils.accuracy(model_ng, val_loader, device)
            train_accuracy = utils.accuracy(model_ng, retain_loader, device)

            print(f'Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        state = model_ng.state_dict()
        torch.save(state, checkpoint_path)
        print('Saved a checkpoint at {}'.format(checkpoint_path))

        return model_ng, training_time

    def neggrad_plus(original, 
                retain_loader, 
                forget_loader, 
                val_loader, 
                checkpoint_path, 
                alpha = 0.9,
                device='cpu',
                seed=1):
        print ("="*30,"> unlearning by NegGrad+ ...")
        model_ngp = copy.deepcopy(original)
        # Define the loss function with class weighting
        learning_rate = 0.001
        num_epochs = 10
        #criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model_ngp.parameters(), lr=learning_rate)
        #scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        checkpoint_path += f'alpha{alpha}-epochs{num_epochs}-lr{learning_rate}-seed{seed}.pth'
        training_time = 0
        for epoch in range(num_epochs):
            model_ngp.train()
            t1 = time.time()
            for (inputs, targets), (inputs_f, targets_f) in zip(retain_loader, cycle(forget_loader)):
                inputs = inputs.to(device)
                targets = targets.to(device)

                inputs_f = inputs_f.to(device)
                targets_f = targets_f.to(device)

                optimizer.zero_grad()

                outputs = model_ngp(inputs)
                loss1 = criterion(outputs, targets)

                outputs_f = model_ngp(inputs_f)
                loss2 = -1 * criterion(outputs_f, targets_f)

                loss = alpha * loss1 + (1-alpha) * loss2


                loss.backward()
                optimizer.step()


            #scheduler.step()

            t2 = time.time()
            training_time += t2 - t1
            
            val_accuracy = utils.accuracy(model_ngp, val_loader, device)
            train_accuracy = utils.accuracy(model_ngp, retain_loader, device)

            print(f'Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        state = model_ngp.state_dict()
        torch.save(state, checkpoint_path)
        print('Saved a checkpoint at {}'.format(checkpoint_path))

        return model_ngp, training_time

    def scrub(original,
            retain_loader,
            forget_loader,
            val_loader,
            checkpoint_path,
            alpha = 0.9,
            max_steps=3, 
            kd_T = 4,
            device='cpu',
            seed=1):
        
        print ("="*30,"> unlearning by SCRUB ...")
        model_scrub = copy.deepcopy(original)
        teacher = copy.deepcopy(original)
        # Define the loss function with class weighting
        learning_rate = 0.001
        num_epochs = 10
        #criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        criterion = nn.CrossEntropyLoss()
        criterion_kl = DistillKL(T=kd_T)
        optimizer = optim.Adam(model_scrub.parameters(), lr=learning_rate)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        checkpoint_path += f'-maxsteps{max_steps}-alpha{alpha}-epochs{num_epochs}-lr{learning_rate}-seed{seed}.pth'
        teacher.eval()
        training_time = 0
        for epoch in range(num_epochs):
            model_scrub.train()
            if epoch < max_steps:
                t1 = time.time()
                for inputs, targets in forget_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    optimizer.zero_grad()

                    outputs = model_scrub(inputs)
                    outputs_t = teacher(inputs)

                    loss_kl = criterion_kl(outputs, outputs_t)

                    loss = -1 * loss_kl


                    loss.backward()
                    optimizer.step()


                scheduler.step()

                t2 = time.time()
                training_time += t2 - t1


            t1 = time.time()
            for inputs, targets in retain_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()

                outputs = model_scrub(inputs)
                outputs_t = teacher(inputs)

                loss1 = criterion(outputs, targets)
                loss_kl = criterion_kl(outputs, outputs_t)

                loss = alpha * loss1 + (1-alpha) * loss_kl


                loss.backward()
                optimizer.step()


            scheduler.step()

            t2 = time.time()
            training_time += t2 - t1

            val_accuracy = utils.accuracy(model_scrub, val_loader, device)
            train_accuracy = utils.accuracy(model_scrub, retain_loader, device)

            print(f'Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        state = model_scrub.state_dict()
        torch.save(state, checkpoint_path)
        print('Saved a checkpoint at {}'.format(checkpoint_path))

        return model_scrub, training_time

    if __name__ == "__main__":

        # Loading data and creating the loaders
        data_handler = DataProcessor(DATA_PATH, COLS, label=LABEL, cat_cols=CAT_COLS, num_cols=NUM_COLS, filters_path=DELETE_FILTER)
        data = data_handler._read_data()
        all_data, reduced_data, deleted_data = data_handler._delete_data(data)

        train_loader, val_loader, test_loader = data_handler._prepare_data(data, update_params=True, bs=2048, test_frac=0.1, val_frac=0.1)
        forget_loader, _, _ = data_handler._prepare_data(deleted_data, update_params=False, bs=2048, test_frac=0, val_frac=0)
        retain_loader, _, _ = data_handler._prepare_data(reduced_data, update_params=False, bs=2048, test_frac=0, val_frac=0)

        # Creating the model
        if MODEL == 'resnet':
            model = models.ResNet1D(num_blocks=[8, 8, 16], 
                                    num_classes=len(data_handler.classes), 
                                    data_dim=data_handler.train_size[1]).to(DEVICE)
        elif MODEL == 'mlp':
            model = models.MLP(input_size=data_handler.train_size[1], 
                            hidden_size=128, 
                            num_classes=len(data_handler.classes), 
                            dropout_prob=0).to(DEVICE)
        else:
            raise ValueError(f'{MODEL} models are not supported yet!')

        os.makedirs("checkpoints/", exist_ok = True)
        checkpoint_path = f'checkpoints/{DATASET}-{MODEL}-selective-'

        original, o_time = train(model, train_loader, val_loader, checkpoint_path+'original-', DEVICE, SEED)
        retrain, re_time = train(model, retain_loader, val_loader, checkpoint_path+'retrain-', DEVICE, SEED)
        model_ft, ft_time = fine_tune(original, retain_loader, val_loader, checkpoint_path+'finetune-', DEVICE, SEED)
        model_ng, ng_time = neggrad(original, forget_loader, val_loader, checkpoint_path+'neggrad-', DEVICE, SEED)
        model_ngp, ngp_time = neggrad_plus(original, retain_loader, forget_loader, val_loader, checkpoint_path+'neggradplus-', 0.99, DEVICE, SEED)
        model_scrub, sc_time = scrub(original, retain_loader, forget_loader, val_loader, checkpoint_path+'scrub-', 0.99, 3, 4, DEVICE, SEED)
        print ("\n\n=========== all the tests ===========\n")
        
        test_acc, retain_acc, forget_acc = utils.all_tests(original, retain_loader, forget_loader, test_loader, DEVICE)
        print(f'Original:\t test: {test_acc:.4f}\t retain: {retain_acc:.4f}\t forget: {forget_acc:.4f}\t time: {o_time}')

        test_acc, retain_acc, forget_acc = utils.all_tests(retrain, retain_loader, forget_loader, test_loader, DEVICE)
        print(f'Retrain:\t test: {test_acc:.4f}\t retain: {retain_acc:.4f}\t forget: {forget_acc:.4f}\t time: {re_time}')

        test_acc, retain_acc, forget_acc = utils.all_tests(model_ft, retain_loader, forget_loader, test_loader, DEVICE)
        print(f'Fine-tune:\t test: {test_acc:.4f}\t retain: {retain_acc:.4f}\t forget: {forget_acc:.4f}\t time: {ft_time}')

        test_acc, retain_acc, forget_acc = utils.all_tests(model_ng, retain_loader, forget_loader, test_loader, DEVICE)
        print(f'NegGrad:\t test: {test_acc:.4f}\t retain: {retain_acc:.4f}\t forget: {forget_acc:.4f}\t time: {ng_time}')

        test_acc, retain_acc, forget_acc = utils.all_tests(model_ngp, retain_loader, forget_loader, test_loader, DEVICE)
        print(f'NegGrad+:\t test: {test_acc:.4f}\t retain: {retain_acc:.4f}\t forget: {forget_acc:.4f}\t time: {ngp_time}')

        test_acc, retain_acc, forget_acc = utils.all_tests(model_scrub, retain_loader, forget_loader, test_loader, DEVICE)
        print(f'SCRUB:\t test: {test_acc:.4f}\t retain: {retain_acc:.4f}\t forget: {forget_acc:.4f}\t time: {sc_time}')
        
