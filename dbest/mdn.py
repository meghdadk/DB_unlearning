import argparse
import os
import sys
import math
import scipy
import random
import time
import dill
import numpy as np
import pandas as pd
import pandasql as ps
import wandb

import matplotlib.pyplot as plt
from concurrent import futures
from copy import deepcopy
from collections import Counter
from itertools import cycle
from numpy.random import choice
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from scipy import integrate,stats
from math import e

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.multiprocessing import Pool
import category_encoders as ce


import Dataset
from sqlParser import Parser
import utils
from utils import mdn_loss, mse_ce_loss, mse_kd_loss, adjust_learning_rate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MDN(nn.Module):
    """A mixture density network architecture
    The input maps to the parameters of a MoG probability distribution, where
    each Gaussian has O dimensions and diagonal covariance.
    Arguments:
        in_features (int): the number of dimensions in the input
        out_features (int): the number of dimensions in the output
        num_gaussians (int): the number of Gaussians per output dimensions
    Input:
        minibatch (BxD): B is the batch size and D is the number of input
            dimensions.
    Output:
        (pi, sigma, mu) (BxG, BxGxO, BxGxO): B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions for each
            Gaussian. Pi is a multinomial distribution of the Gaussians. Sigma
            is the standard deviation of each Gaussian. Mu is the mean of each
            Gaussian.
    """

    def __init__(self, in_features, out_features, num_gaussians):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.pi = nn.Sequential(
            nn.Linear(in_features, num_gaussians)
        )

        self.sigma = nn.Linear(in_features, out_features * num_gaussians)
        self.mu = nn.Linear(in_features, out_features * num_gaussians)

    def forward(self, minibatch):
        pi = self.pi(minibatch)
        sigma = torch.exp(self.sigma(minibatch))
        #elu = nn.ELU()
        #sigma = elu(self.sigma(minibatch)) + 1 + 1e-7
        sigma = sigma.view(-1, self.num_gaussians, self.out_features)
        mu = self.mu(minibatch)
        mu = mu.view(-1, self.num_gaussians, self.out_features)
        return pi, sigma, mu

class DenMDN:
    """A Mixture Density Network Class for Density Estimation"""

    def __init__(
            self,
            dataset,
            args,
            b_normalize_data=True):

        self.model = None
        self.b_normalize_data = b_normalize_data
        self.batch_size = None
        self.n_epoch = None
        self.dataset = dataset
        self.n_hidden_layer=args.num_hid_layers
        self.n_hidden_nodes=args.hid_layer_sizes
        self.n_gaussians=args.num_gaussians

        
    def make_model(self,input_dim):
        """Initializing a Mixture Density Network 

        Args:
            n_hidden_layers (integer): number of hidden layers. Currently 1,2 or 5. 
            n_hidden_nodes (list-like): a list of the number of neurons in each hidden layer. len(n_hidden_nodes) = n_hidden_layers
            n_gaussians (integer): number of gaussian components of the MDN.
        """ 


        # initialize the model

        if self.n_hidden_layer == 1:
            self.model = nn.Sequential(
                nn.Linear(input_dim, self.n_hidden_nodes[0]),
                nn.ReLU(),
                nn.Dropout(0.2),
                MDN(self.n_hidden_nodes[0], 1, self.n_gaussians),
            )
        elif self.n_hidden_layer == 2:
            self.model = nn.Sequential(
                nn.Linear(input_dim, self.n_hidden_nodes[0]),
                nn.ReLU(),
                nn.Linear(self.n_hidden_nodes[0], self.n_hidden_nodes[1]),
                nn.ReLU(),
                nn.Dropout(0.2),
                MDN(self.n_hidden_nodes[1], 1, self.n_gaussians),
            )
        elif self.n_hidden_layer == 5:
            self.model = nn.Sequential(
                nn.Linear(input_dim, self.n_hidden_nodes[0]),
                nn.ReLU(),
                nn.Linear(self.n_hidden_nodes[0], self.n_hidden_nodes[1]),
                nn.ReLU(),
                nn.Linear(self.n_hidden_nodes[1], self.n_hidden_nodes[2]),
                nn.ReLU(),
                nn.Linear(self.n_hidden_nodes[2], self.n_hidden_nodes[3]),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.n_hidden_nodes[3], self.n_hidden_nodes[4]),
                nn.ReLU(),
                nn.Dropout(0.2),
                MDN(self.n_hidden_nodes[4], 1, self.n_gaussians),
            )
        else:
            raise ValueError(
                "The hidden layer should be 1, 2, or 5 but you provided "
                + str(self.n_hidden_layer)
            )

    def fit(
            self,
            x_points,
            y_points,
            args,
            encoded=True,
            n_workers=0,
            logger=None): 

        """ Fitting a mixture density network to a (x,y) dataset. x is the input and y is the target variable

        Args:
            x_points (list-like): training X values
            y_points(list-like): Target values corresponding to 'x_points'
            encoded (Boolian): Whether the X values are categorical and have been encoded or not. 
            lr: learning rate
            n_epoch: number of epochs
            batch_size: batch size of the data loader
            n_workers: for torch data loader

        """

        self.lr = args.learning_rate
        self.batch_size = args.batch_size
        self.n_epochs = args.epochs
        self.device = device


        tensor_xs = torch.from_numpy(x_points.astype(np.float32)) 
        y_points = np.asarray(y_points).reshape(-1,1)
        tensor_ys = torch.from_numpy(y_points.astype(np.float32))
         


        tensor_xs = tensor_xs
        tensor_ys = tensor_ys
        print (x_points.shape, y_points.shape)
        print (tensor_xs.shape,tensor_ys.shape)

        train_kwargs = {'batch_size': self.batch_size}
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}

        train_kwargs.update(cuda_kwargs)


        my_dataset = torch.utils.data.TensorDataset(
            tensor_xs, tensor_ys
        )  
        my_dataloader = torch.utils.data.DataLoader(
            my_dataset,
            **train_kwargs
        )

        if encoded:
            input_dim = len((list(self.dataset.encoders.values())[0].categories_[0]))
        else:
            input_dim = 1

        
        self.make_model(input_dim=input_dim)

        self.model = self.model.to(device)

        if args.opt == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif args.opt == "sgd":
            optimizer = optim.SGD(self.model.parameters(),
                                  lr=self.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)
        elif args.opt == "rmsp":
            optimizer = optim.RMSprop(self.model.parameters(),
                                  lr=self.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)
        else:
            raise NotImplementedError(args.opt)

        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=0.98
        )


        if args.evaluate:
            print ("setting up evaluation benchmark")
            import benchmarking
            for agg in args.aggs:
                benchmarking.create_the_benchmark(args, args.dataset, datafile=args.datafile,
                                                  benchmark_dir='benchmark/', cat_att=args.x_att,
                                                  range_att=args.y_att, sep=',', agg=agg, num_queries=args.num_eval_queries)


        criteria = mdn_loss
        self.model.train()

        if isinstance(logger, type(wandb.run)) and logger is not None:
            logger.name = args.dataset+"_"+args.mode + "_lr{}".format(self.lr) + "_bs{}".format(self.batch_size) + \
                             "_{}".format((type (optimizer).__name__)) + "_q{}".format(args.num_eval_queries)
            logger.watch(self.model)

        lr = args.learning_rate
        for epoch in range(1, self.n_epochs+1):
            losses = []
            adjust_learning_rate(epoch, args, optimizer)
            for minibatch, labels in my_dataloader:
                minibatch = minibatch.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                pi, sigma, mu = self.model(minibatch)
                loss = criteria(pi, sigma, mu, labels, device)
                loss.backward()
                losses.append(loss.item())
                optimizer.step()

            lr = optimizer.param_groups[0]['lr']
            if epoch % 1 == 0:
                    print("Epoch {}\t loss {:.4f}\t lr {:.4f}".format(epoch, np.mean(losses), lr))


            self.eval(epoch, lr, np.mean(losses), logger, args, None, None, None, None)


        self.eval(0, lr, None, logger, args, None, None, None, None)


        self.model.eval()
        print("Finish regression training.")
        return self

    def finetune(
            self,
            x_points,
            y_points,
            x_deletion,
            y_deletion,
            args,
            encoded=True,
            n_workers=0,
            keep_weights=True,
            logger=None):

        """ fine-tuning a pre-trained model

        Args:
            x_points (list-like): training X values
            y_points(list-like): Target values corresponding to 'x_points'
            encoded (Boolian): Whether the X values are categorical and have been encoded or not. 
            lr: learning rate
            n_epoch: number of epochs
            batch_size: batch size of the data loader
            n_workers: for torch data loader

        """

        self.lr = args.learning_rate
        self.batch_size = args.batch_size
        self.n_epochs = args.epochs
        self.device = device


        tensor_xs = torch.from_numpy(x_points.astype(np.float32)) 
        y_points = np.asarray(y_points).reshape(-1,1)
        tensor_ys = torch.from_numpy(y_points.astype(np.float32))


        tensor_xs_del = torch.from_numpy(x_deletion.astype(np.float32)) 
        y_transfer_del = np.asarray(y_deletion).reshape(-1,1)
        tensor_ys_del = torch.from_numpy(y_transfer_del.astype(np.float32))


        my_dataset = torch.utils.data.TensorDataset(
            tensor_xs, tensor_ys
        )
        print ("training size:", len(my_dataset))
        my_dataloader = torch.utils.data.DataLoader(
            my_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,
            num_workers=1
        )

        if encoded:
            input_dim = len((list(self.dataset.encoders.values())[0].categories_[0]))
        else:
            input_dim = 1

        if not keep_weights:
            self.make_model(input_dim=input_dim)


        self.model = self.model.to(device)

        if args.opt == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif args.opt == "sgd":
            optimizer = optim.SGD(self.model.parameters(),
                                  lr=self.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)
        elif args.opt == "rmsp":
            optimizer = optim.RMSprop(self.model.parameters(),
                                  lr=self.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)
        else:
            raise NotImplementedError(args.opt)

        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=0.98
        )



        if args.evaluate and not args.use_pre_queries:
            print ("setting up evaluation benchmark")
            import benchmarking
            for agg in args.aggs:
                benchmarking.create_the_benchmark(args, dataset=args.dataset, datafile=args.datafile.replace('.csv','_reduced.csv'),
                                                  filters_path=args.filters, benchmark_dir='benchmark/', cat_att=args.x_att,
                                                  range_att=args.y_att, num_queries=args.num_eval_queries, sep=',', agg=agg)



        criteria = utils.mdn_loss
        self.model.train()
        lr = args.learning_rate
        
        if isinstance(logger, type(wandb.run)) and logger is not None:
            logger.name = args.dataset+"_"+args.mode + "_lr{}".format(self.lr) + "_bs{}".format(self.batch_size) + \
                             "_{}".format((type (optimizer).__name__)) + "_q{}".format(args.num_eval_queries) + "_lendel{}".format(len(tensor_xs_del))

            logger.config.update({"len deleted data": len(tensor_xs_del)})
            logger.watch(self.model)
        
        for epoch in range(1, self.n_epochs+1):
            losses = []
            adjust_learning_rate(epoch, args, optimizer)
            for minibatch, labels in my_dataloader:
                minibatch = minibatch.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                pi, sigma, mu = self.model(minibatch)
                loss = criteria(pi, sigma, mu, labels, device)
                loss.backward()
                losses.append(loss.item())
                optimizer.step()
            
            #if args.opt == "adam":
            #    my_lr_scheduler.step()
            lr = optimizer.param_groups[0]['lr']
            if epoch % 1 == 0:
                    print("Epoch {}\t loss {:.4f}\t lr{:.4f}".format(epoch, np.mean(losses), lr))

            self.eval(epoch, lr, np.mean(losses), logger, args, tensor_xs, tensor_ys, tensor_xs_del, tensor_ys_del)


        self.eval(0,lr,  None, logger, args, tensor_xs, tensor_ys, tensor_xs_del, tensor_ys_del)
        self.model.eval()
        return self

    def scrub(
            self,
            x_points,
            y_points,
            x_deletion,
            y_deletion,
            args,
            keep_weights=True,
            encoded=True,
            n_workers=0,
            logger=None): 

        """ Deleting (x_deletion,y_deletion) from the model \
            The intersection between transfer-set and \
            deletion-set must be empty.
            Here, we have two inner loops for training \
            In the first loop we maximize loss_bad \
            in the inner loop we minimize loss_good

        Args:
            x_points (list-like): retain data X values
            y_transfer (list-like): Target values corresponding to 'x_points'
            x_deletion (list-like): X values to be deleted
            y_deletion (list-like): y values corresponding to 'x_deletion'
            keep_weights (Boolian): Whether to keep the current weights of the model or \
                                    building it from scratch
            encoded (Boolian): Whether the X values are categorical and have been encoded or not. 
            lr: learning rate
            n_epoch: number of epochs
            batch_size: batch size of the data loader
            n_workers: for torch data loader

        """

        self.lr = args.learning_rate
        self.n_epochs= args.epochs
        self.batch_size = args.batch_size
        self.device = device


        tensor_xs = torch.from_numpy(x_points.astype(np.float32)) 
        y_points = np.asarray(y_points).reshape(-1,1)
        tensor_ys = torch.from_numpy(y_points.astype(np.float32))
         

        tensor_xs_del = torch.from_numpy(x_deletion.astype(np.float32)) 
        y_transfer_del = np.asarray(y_deletion).reshape(-1,1)
        tensor_ys_del = torch.from_numpy(y_transfer_del.astype(np.float32))


        my_dataset = torch.utils.data.TensorDataset(
            tensor_xs, tensor_ys
        )
        print ("training size:", len(my_dataset))
        my_dataloader = torch.utils.data.DataLoader(
            my_dataset,
            batch_size=args.batch_size,
            pin_memory=True,
            shuffle=True,
            num_workers=1
        )


        deletion_dataset = torch.utils.data.TensorDataset(
            tensor_xs_del, tensor_ys_del
        )
        deletion_dataloader = torch.utils.data.DataLoader(
            deletion_dataset,
            batch_size=args.del_batch_size,
            pin_memory=True,
            shuffle=True,
            num_workers=1
        )

        if encoded:
            input_dim = len((list(self.dataset.encoders.values())[0].categories_[0]))
        else:
            input_dim = 1
       

        good_teacher_model = deepcopy(self.model).to(device)
        bad_teacher_model = deepcopy(self.model).to(device)

        if not keep_weights:
            self.make_model(input_dim=input_dim)

        self.model = self.model.to(device)


        if args.opt == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif args.opt == "sgd":
            optimizer = optim.SGD(self.model.parameters(),
                                  lr=self.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)
        elif args.opt == "rmsp":
            optimizer = optim.RMSprop(self.model.parameters(),
                                  lr=self.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)
        else:
            raise NotImplementedError(args.opt)

        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=0.98
        )


        if args.evaluate and not args.use_pre_queries:
            print ("setting up evaluation benchmark")
            import benchmarking
            for agg in args.aggs:
                benchmarking.create_the_benchmark(args, dataset=args.dataset, datafile=args.datafile.replace('.csv','_reduced.csv'),
                                                  filters_path=args.filters, benchmark_dir='benchmark/', cat_att=args.x_att,
                                                  range_att=args.y_att, num_queries=args.num_eval_queries, sep=',', agg=agg)



        bad_kd_loss = 0
        mdn_criteria = mdn_loss
        kd_criteria = mse_ce_loss
        self.model.train()
        lr = args.learning_rate

        if isinstance(logger, type(wandb.run)) and logger is not None:
            logger.name = args.dataset+"_"+args.mode + "_lr{}".format(self.lr) + "_bs{}".format(self.batch_size) + \
                             "_delbs{}".format(args.del_batch_size) + "_{}".format((type (optimizer).__name__)) + \
                             "_q{}".format(args.num_eval_queries) + "_lendel{}".format(len(tensor_xs_del))
            logger.watch(self.model)
            logger.config.update({"len deleted data": len(tensor_xs_del)})

        for epoch in range(1, self.n_epochs+1):
            losses = []
            adjust_learning_rate(epoch, args, optimizer)

            for minibatch, labels in my_dataloader:
                minibatch = minibatch.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                pi_s, sigma_s, mu_s = self.model(minibatch) #forwarding the original data through the student model
                pi_gt, sigma_gt, mu_gt = good_teacher_model(minibatch)  #forwarding the original data through the good teacher

                reg_loss = mdn_criteria(pi_s, sigma_s, mu_s, labels, device) #regular nll loss for the student model
                kd_loss = kd_criteria(pi_gt, sigma_gt, mu_gt, pi_s, sigma_s, mu_s, labels, device, T=1) #distillation loss between good teacher and student

                loss =  args.alpha*reg_loss + (1-args.alpha)*kd_loss 


                loss.backward()
                optimizer.step()

            if epoch <= args.msteps:
                for minibatch_del, labels_del in deletion_dataloader:
                    minibatch_del = minibatch_del.to(device)
                    labels_del = labels_del.to(device)

                    optimizer.zero_grad()

                    pi_s_del, sigma_s_del, mu_s_del = self.model(minibatch_del)     #forwarding the deletion-set through the student model

                    pi_bt, sigma_bt, mu_bt = bad_teacher_model(minibatch_del)  #forwarding the deletion set through the bad teacher
                    
                    neg_kd_loss = -kd_criteria(pi_bt, sigma_bt, mu_bt, pi_s_del, sigma_s_del, mu_s_del, labels_del, device, T=1) #distillation loss between bad teacher and student
                    neg_reg_loss = -mdn_criteria(pi_s_del, sigma_s_del, mu_s_del, labels_del, device)

                    loss = neg_kd_loss#args.beta*neg_reg_loss + (1-args.beta)*neg_kd_loss
                    loss.backward()

                    """
                    try:
                        assert ~self.model[3].pi[0].weight.grad.isnan().any()
                        assert ~self.model[3].sigma.weight.grad.isnan().any()
                        assert ~self.model[3].mu.weight.grad.isnan().any()
                    except:
                        print(epoch, i)
                        print (self.model[3].pi[0].weight.grad)
                        print (self.model[3].sigma.weight.grad)
                        print (self.model[3].mu.weight.grad)
                        print (self.model[3].pi[0].weight)
                        print (self.model[3].sigma.weight)
                        print (self.model[3].mu.weight)
                        os.sys.exit()
                    """
                    optimizer.step()


            for minibatch, labels in my_dataloader:
                minibatch = minibatch.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                pi_s, sigma_s, mu_s = self.model(minibatch) #forwarding the original data through the student model
                pi_gt, sigma_gt, mu_gt = good_teacher_model(minibatch)  #forwarding the original data through the good teacher

                reg_loss = mdn_criteria(pi_s, sigma_s, mu_s, labels, device) #regular nll loss for the student model
                kd_loss = kd_criteria(pi_gt, sigma_gt, mu_gt, pi_s, sigma_s, mu_s, labels, device, T=1) #distillation loss between good teacher and student
                loss =  args.alpha*reg_loss + (1-args.alpha)*kd_loss 
                loss.backward()
                losses.append(loss.item())
                optimizer.step()

            lr = optimizer.param_groups[0]['lr']
            if epoch % 1 == 0:
                    print("Epoch {}\t loss {:.4f}\t lr{:.4f}".format(epoch, np.mean(losses), lr))
            
            self.eval(epoch, lr, np.mean(losses), logger, args, tensor_xs, tensor_ys, tensor_xs_del, tensor_ys_del)


        self.eval(0, lr, None, logger, args, tensor_xs, tensor_ys, tensor_xs_del, tensor_ys_del)


        self.model.eval()
        return self

    def negativegrad(
            self,
            x_points,
            y_points,
            x_deletion,
            y_deletion,
            args,
            keep_weights=True,
            encoded=True,
            n_workers=0,
            logger=None): 

        """ Deleting (x_deletion,y_deletion) from the model \
            The intersection between transfer-set and \
            deletion-set must be empty.
            Here, we have two inner loops for training \
            In the first loop we maximize loss_bad \
            in the inner loop we minimize loss_good

        Args:
            x_points (list-like): retain data X values
            y_transfer (list-like): Target values corresponding to 'x_points'
            x_deletion (list-like): X values to be deleted
            y_deletion (list-like): y values corresponding to 'x_deletion'
            keep_weights (Boolian): Whether to keep the current weights of the model or \
                                    building it from scratch
            encoded (Boolian): Whether the X values are categorical and have been encoded or not. 
            lr: learning rate
            n_epoch: number of epochs
            batch_size: batch size of the data loader
            n_workers: for torch data loader

        """

        self.lr = args.learning_rate
        self.n_epochs= args.epochs
        self.batch_size = args.batch_size
        self.device = device


        tensor_xs = torch.from_numpy(x_points.astype(np.float32)) 
        y_points = np.asarray(y_points).reshape(-1,1)
        tensor_ys = torch.from_numpy(y_points.astype(np.float32))
         

        tensor_xs_del = torch.from_numpy(x_deletion.astype(np.float32)) 
        y_transfer_del = np.asarray(y_deletion).reshape(-1,1)
        tensor_ys_del = torch.from_numpy(y_transfer_del.astype(np.float32))


        my_dataset = torch.utils.data.TensorDataset(
            tensor_xs, tensor_ys
        )
        print ("training size:", len(my_dataset))
        my_dataloader = torch.utils.data.DataLoader(
            my_dataset,
            batch_size=args.batch_size,
            pin_memory=True,
            shuffle=True,
            num_workers=1
        )


        deletion_dataset = torch.utils.data.TensorDataset(
            tensor_xs_del, tensor_ys_del
        )
        deletion_dataloader = torch.utils.data.DataLoader(
            deletion_dataset,
            batch_size=args.del_batch_size,
            pin_memory=True,
            shuffle=True,
            num_workers=1
        )

        if encoded:
            input_dim = len((list(self.dataset.encoders.values())[0].categories_[0]))
        else:
            input_dim = 1
       

        if not keep_weights:
            self.make_model(input_dim=input_dim)

        self.model = self.model.to(device)


        if args.opt == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif args.opt == "sgd":
            optimizer = optim.SGD(self.model.parameters(),
                                  lr=self.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)
        elif args.opt == "rmsp":
            optimizer = optim.RMSprop(self.model.parameters(),
                                  lr=self.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)
        else:
            raise NotImplementedError(args.opt)

        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=0.98
        )


        if args.evaluate and not args.use_pre_queries:
            print ("setting up evaluation benchmark")
            import benchmarking
            for agg in args.aggs:
                benchmarking.create_the_benchmark(args, dataset=args.dataset, datafile=args.datafile.replace('.csv','_reduced.csv'),
                                                  filters_path=args.filters, benchmark_dir='benchmark/', cat_att=args.x_att,
                                                  range_att=args.y_att, num_queries=args.num_eval_queries, sep=',', agg=agg)


        bad_kd_loss = 0
        mdn_criteria = mdn_loss
        kd_criteria = mse_kd_loss
        self.model.train()

        if isinstance(logger, type(wandb.run)) and logger is not None:
            logger.name = args.dataset+"_"+args.mode + "_lr{}".format(self.lr) + "_bs{}".format(self.batch_size) + \
                             "_delbs{}".format(args.del_batch_size) + "_{}".format((type (optimizer).__name__)) + \
                             "_q{}".format(args.num_eval_queries) + "_lendel{}".format(len(tensor_xs_del))
        
            logger.watch(self.model)
            logger.config.update({"len deleted data": len(tensor_xs_del)})

        lr = args.learning_rate
        for epoch in range(1, self.n_epochs+1):
            losses = []
            adjust_learning_rate(epoch, args, optimizer)
            if len(my_dataloader) >= len(deletion_dataloader):
                for (minibatch, labels), (minibatch_del, labels_del) in zip(my_dataloader, cycle(deletion_dataloader)):
                    minibatch = minibatch.to(device)
                    labels = labels.to(device)
                    minibatch_del = minibatch_del.to(device)
                    labels_del = labels_del.to(device)

                    optimizer.zero_grad()

                    pi_s, sigma_s, mu_s = self.model(minibatch) #forwarding the original data through the student model
                    pi_s_del, sigma_s_del, mu_s_del = self.model(minibatch_del)     #forwarding the deletion-set through the student model

                    reg_loss = mdn_criteria(pi_s, sigma_s, mu_s, labels, device) #regular nll loss for the student model
                    neg_reg_loss = -mdn_criteria(pi_s_del, sigma_s_del, mu_s_del, labels_del, device)

                    loss =  args.alpha*reg_loss + (1-args.alpha)*neg_reg_loss 

                    loss.backward()
                    losses.append(loss.item())
                    optimizer.step()
            else:
                for (minibatch_del, labels_del), (minibatch, labels) in zip(deletion_dataloader, cycle(my_dataloader)):
                    minibatch = minibatch.to(device)
                    labels = labels.to(device)
                    minibatch_del = minibatch_del.to(device)
                    labels_del = labels_del.to(device)

                    optimizer.zero_grad()

                    pi_s, sigma_s, mu_s = self.model(minibatch) 
                    pi_s_del, sigma_s_del, mu_s_del = self.model(minibatch_del)   

                    reg_loss = mdn_criteria(pi_s, sigma_s, mu_s, labels, device) 
                    neg_reg_loss = -mdn_criteria(pi_s_del, sigma_s_del, mu_s_del, labels_del, device)

                    loss =  args.alpha*reg_loss + (1-args.alpha)*neg_reg_loss
                    losses.append(loss.item())

                    loss.backward()
                    optimizer.step()                

            lr = optimizer.param_groups[0]['lr']
            if epoch % 1 == 0:
                    print("Epoch {}\t loss {:.4f}\t lr{:.4f}".format(epoch, np.mean(losses), lr))
            
            self.eval(epoch, lr, np.mean(losses), logger, args, tensor_xs, tensor_ys, tensor_xs_del, tensor_ys_del)


        self.eval(0, lr, None, logger, args, tensor_xs, tensor_ys, tensor_xs_del, tensor_ys_del)


        self.model.eval()
        return self

    def predict(
            self,
            x_points,
            y_points,
            encoded=True):
    
        """Predicting the Gaussian probabilities (densities)

        Args:
            x_points (list-like): predicting X values
            y_points (list-like): y values corresponding to x_points


        Output:
            results (1-d array): array of the predicted densities
        """


        tensor_xs = torch.from_numpy(x_points.astype(np.float32)) 
        y_points = np.asarray(y_points).reshape(-1,1)
         

        # move variables to cuda
        tensor_xs = tensor_xs.to(device)


        if encoded:
            input_dim = len((list(self.dataset.encoders.values())[0].categories_[0]))
        else:
            input_dim = 1
        
        softmax = nn.Softmax(dim=1)
        pis, sigmas, mus = self.model(tensor_xs)
        pis = softmax(pis)
        pis = pis.cpu()
        sigmas = sigmas.cpu()
        mus = mus.cpu()

        #pis = torch.cat(y_points.shape[0] * [pis])
        #mus = torch.cat(y_points.shape[0] * [mus])
        #sigmas = torch.cat(y_points.shape[0] * [sigmas])
        
        

        mus = mus.squeeze().detach().numpy().reshape(1,-1)
        pis = pis.squeeze().detach().numpy().reshape(1,-1)
        sigmas = sigmas.squeeze().detach().numpy().reshape(1,-1)
        #print (pis.shape, mus.shape, sigmas.shape, y_points.shape)

        result = np.array(
            [
                np.multiply(stats.norm(mus, sigmas).pdf(y), pis)
                .sum(axis=1)
                .tolist()
                for y in y_points
            ]
        ).transpose()
        #noises = result < np.max(result)/7
        #result[noises] = 0
        return result

    def eval(
            self,
            epoch,
            lr, 
            loss,
            logger,
            args,
            tensor_xs,
            tensor_ys,
            tensor_xs_del,
            tensor_ys_del):
        import benchmarking

        if args.evaluate and epoch % args.eval_per_epoch == 0:
            for agg in args.aggs:
                if agg == 'count':
                    metric = 'absolute-error'
                else:
                    metric = 'absolute-error'
                metrics_retain, queries, reals, predictions = benchmarking.run(model=self, queries_file="benchmark/{}_{}_queries.sql".format(args.dataset,agg),
                    ground_truth="benchmark/{}_{}_queries.csv".format(args.dataset,agg), metric=metric)
                if epoch == 0:
                    if logger is not None:
                        logger.log({
                                f"final-mean-{agg}": metrics_retain["mean"],
                                f"final-median-{agg}": metrics_retain["median"], 
                                f"final-90th-percentile-{agg}": metrics_retain["90th"],
                                f"final-95th-percentile-{agg}": metrics_retain["95th"], 
                                f"final-99th-percentile-{agg}": metrics_retain["99th"]})
                        logger.config.update({f"queries-{agg}": queries,
                                   f"ground-truth-{agg}": reals,
                                   f"predictions-{agg}": predictions})
                else:
                    if logger is not None:
                        logger.log({"loss": loss,
                                "lr": lr, 
                                f"mean-err-{agg}": metrics_retain["mean"],
                                f"median-err-{agg}": metrics_retain["median"], 
                                f"90th-percentile-{agg}": metrics_retain["90th"],
                                f"95th-percentile-{agg}": metrics_retain["95th"], 
                                f"99th-percentile-{agg}": metrics_retain["99th"]})

                if args.mode != "train" and args.eval_deleted == True:
                    metrics_deleted, queries_deleted, reals_deleted, predictions_deleted = benchmarking.run(model=self, queries_file="benchmark/{}_{}_deleted_rows_query.sql".format(args.dataset,agg),
                        ground_truth="benchmark/{}_{}_deleted_rows_query.csv".format(args.dataset,agg), metric=metric)
                    #metrics_both, queries_both, reals_both, predictions_both = benchmarking.run(model=self, queries_file="benchmark/{}_{}_both_rows_query.sql".format(args.dataset,agg),
                    #    ground_truth="benchmark/{}_{}_both_rows_query.csv".format(args.dataset,agg), metric=metric)
                    if epoch == 0:
                        if logger is not None:
                            logger.log({
                                    f"final-mean-deleted-err-{agg}": metrics_deleted["mean"],
                                    f"final-median-deleted-err-{agg}": metrics_deleted["median"],
                                    f"final-95th-deleted-err-{agg}": metrics_deleted["95th"],
                                    f"final-99th-deleted-err-{agg}": metrics_deleted["99th"]
                                    })
                        logger.config.update({f"queries-deleted-{agg}": queries_deleted,
                                   f"ground-truth-deleted-{agg}": reals_deleted,
                                   f"predictions-deleted-{agg}": predictions_deleted})
                    else:
                        if logger is not None:
                            logger.log({
                                    f"mean-deleted-err-{agg}": metrics_deleted["mean"],
                                    f"median-deleted-error-{agg}": metrics_deleted["median"],
                                    f"95th-deleted-err-{agg}": metrics_deleted["95th"],
                                    f"99th-deleted-error-{agg}": metrics_deleted["99th"]
                                    })

        if args.eval_likelihood and epoch % args.eval_per_epoch == 0 and args.mode != "train":
            batch = tensor_xs_del.to(device)
            labels = tensor_ys_del.to(device)
            pi_s, sigma_s, mu_s = self.model(batch)
            likelihood_del = mdn_loss(pi_s, sigma_s, mu_s, labels, device, _return='likelihood')

            batch = tensor_xs.to(device)
            labels = tensor_ys.to(device)
            pi_s, sigma_s, mu_s = self.model(batch)
            likelihood_retain = mdn_loss(pi_s, sigma_s, mu_s, labels, device, _return='likelihood')

            if epoch == 0:
                if logger is not None:
                    logger.log({"final-del-likelihood":likelihood_del})
                    logger.log({"final-retain-likelihood":likelihood_retain})
            else:
                if logger is not None:
                    logger.log({"deleted_likelihood": likelihood_del})
                    logger.log({"retain_likelihood": likelihood_retain})

        if args.compare_hist and epoch == 0:
            if args.mode == 'train':
                datafile = args.datafile
                len_del = ""
            else:
                datafile = args.datafile.replace('.csv', '_reduced.csv')
                len_del = str(len(tensor_xs_del))
            real, generated, path_r, path_s = benchmarking.plot_histogram(self, datafile, args.x_att, args.y_att, args.x_val_for_hist, args.dataset+"_"+args.mode+"_"+len_del, args.mode, args.bins)
            if logger is not None:
                #logger.log({"real data sample": real})
                #logger.log({"generated data": generated})
                logger.save(path_r)
                logger.save(path_s)
        
        else:
            if logger is not None:
                logger.log({"loss": loss, "lr": lr})