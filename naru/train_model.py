"""Model training."""
import argparse
import json
import os
from copy import deepcopy
import time
import subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import cycle
import shutil
import wandb

import common
import datasets
import made
import transformer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATASETS = ['dmv-tiny', 'dmv', 'census', 'forest']

print('Device', DEVICE)
#os.environ["WANDB_MODE"] = "offline"


def parser():
    parser = argparse.ArgumentParser()

    # Training.
    parser.add_argument('--mode', type=str, default='train', help='train or unlearn.')
    parser.add_argument('--pre-model', type=str, default=None, help='fine-tune on a pre-trained model.')
    parser.add_argument('--remove-weights', action='store_true', help='re-initialize the weights of the model?')
    parser.add_argument('--dataset', type=str, default='dmv-tiny', help='Dataset.')
    parser.add_argument('--filters', type=str, default='filters.json', help='the json file containint the filters to remove rows')
    parser.add_argument('--wandb-save', action='store_true', help='save model and workload to wandb cloud?')
    parser.add_argument('--eval-per-epoch', type=int, help='the number i means evaluate every ith epoch. 0 means disabled')
    parser.add_argument('--eval-deleted', action='store_true', help='whether to evaluate deleted parts')
    parser.add_argument('--num-gpus', type=int, default=0, help='#gpus.')
    parser.add_argument('--bs', type=int, default=1024, help='Batch size.')
    parser.add_argument('--del-bs', type=int, default=1024, help='delete set batch size')
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha parameter')
    parser.add_argument('--msteps', type=int, default=5, help='maximum number of steps in the ascent direction')
    parser.add_argument('--tr-frac', type=float, default=1, help='transfer-set size')
    parser.add_argument('--learning-rate', type=float, required=True, help='start learning rate')
    parser.add_argument(
        '--warmups',
        type=int,
        default=0,
        help='Learning rate warmup steps.  Crucial for Transformer.')
    parser.add_argument('--epochs',
                        type=int,
                        default=20,
                        help='Number of epochs to train for.')
    parser.add_argument('--constant-lr',
                        type=float,
                        default=None,
                        help='Constant LR?')
    parser.add_argument(
        '--column-masking',
        action='store_true',
        help='Column masking training, which permits wildcard skipping'\
        ' at querying time.')

    # MADE.
    parser.add_argument('--fc-hiddens',
                        type=int,
                        default=128,
                        help='Hidden units in FC.')
    parser.add_argument('--layers', type=int, default=4, help='# layers in FC.')
    parser.add_argument('--residual', action='store_true', help='ResMade?')
    parser.add_argument('--direct-io', action='store_true', help='Do direct IO?')
    parser.add_argument(
        '--inv-order',
        action='store_true',
        help='Set this flag iff using MADE and specifying --order. Flag --order '\
        'lists natural indices, e.g., [0 2 1] means variable 2 appears second.'\
        'MADE, however, is implemented to take in an argument the inverse '\
        'semantics (element i indicates the position of variable i).  Transformer'\
        ' does not have this issue and thus should not have this flag on.')
    parser.add_argument(
        '--input-encoding',
        type=str,
        default='binary',
        help='Input encoding for MADE/ResMADE, {binary, one_hot, embed}.')
    parser.add_argument(
        '--output-encoding',
        type=str,
        default='one_hot',
        help='Iutput encoding for MADE/ResMADE, {one_hot, embed}.  If embed, '
        'then input encoding should be set to embed as well.')

    # Transformer.
    parser.add_argument(
        '--heads',
        type=int,
        default=0,
        help='Transformer: num heads.  A non-zero value turns on Transformer'\
        ' (otherwise MADE/ResMADE).'
    )
    parser.add_argument('--blocks',
                        type=int,
                        default=2,
                        help='Transformer: num blocks.')
    parser.add_argument('--dmodel',
                        type=int,
                        default=32,
                        help='Transformer: d_model.')
    parser.add_argument('--dff', type=int, default=128, help='Transformer: d_ff.')
    parser.add_argument('--transformer-act',
                        type=str,
                        default='gelu',
                        help='Transformer activation.')

    # Ordering.
    parser.add_argument('--num-orderings',
                        type=int,
                        default=1,
                        help='Number of orderings.')
    parser.add_argument(
        '--order',
        nargs='+',
        type=int,
        required=False,
        help=
        'Use a specific ordering.  '\
        'Format: e.g., [0 2 1] means variable 2 appears second.'
    )


    #Wandb arguments
    parser.add_argument('--wandb_mode', type=str, default='disabled', choices=['online', 'offline', 'disabled'], 
                        help='wandb running mode')
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='the project on wandb to add the runs')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='your wandb user name')
    parser.add_argument('--wandb_run_id', type=str, default=None,
                        help='To resume a previous run with an id')
    parser.add_argument('--wandb_group_name', type=str, default=None,
                        help='Given name to group runs together')


    args = parser.parse_args()

    return args

def init_logger(args):

    if args.wandb_group_name is None:
        args.wandb_group_name = args.dataset
    if args.wandb_run_id is not None:
        logger = wandb.init(id=args.wandb_run_id, resume="must")
    else:
        logger = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                   mode=args.wandb_mode, group=args.wandb_group_name, config=args)

    return logger


def Entropy(name, data, bases=None):
    import scipy.stats
    s = 'Entropy of {}:'.format(name)
    ret = []
    for base in bases:
        assert base == 2 or base == 'e' or base is None
        e = scipy.stats.entropy(data, base=base if base != 'e' else None)
        ret.append(e)
        unit = 'nats' if (base == 'e' or base is None) else 'bits'
        s += ' {:.4f} {}'.format(e, unit)
    print(s)
    return ret


def RunEpoch(split,
             model,
             opt,
             train_data,
             val_data=None,
             batch_size=100,
             upto=None,
             epoch_num=None,
             verbose=False,
             log_every=10,
             return_losses=False,
             table_bits=None):
    torch.set_grad_enabled(split == 'train')
    model.train() if split == 'train' else model.eval()
    dataset = train_data if split == 'train' else val_data
    losses = []

    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=(split == 'train'))

    # How many orderings to run for the same batch?
    nsamples = 1
    if hasattr(model, 'orderings'):
        nsamples = len(model.orderings)

    for step, xb in enumerate(loader):
        if split == 'train':
            base_lr = 8e-4
            for param_group in opt.param_groups:
                if args.constant_lr:
                    lr = args.constant_lr
                elif args.warmups:
                    t = args.warmups
                    d_model = model.embed_size
                    global_steps = len(loader) * epoch_num + step + 1
                    lr = (d_model**-0.5) * min(
                        (global_steps**-.5), global_steps * (t**-1.5))
                else:
                    lr = 1e-2

                param_group['lr'] = lr

        if upto and step >= upto:
            break

        xb = xb.to(DEVICE).to(torch.float32)

        # Forward pass, potentially through several orderings.
        xbhat = None
        model_logits = []
        num_orders_to_forward = 1
        if split == 'test' and nsamples > 1:
            # At test, we want to test the 'true' nll under all orderings.
            num_orders_to_forward = nsamples

        for i in range(num_orders_to_forward):
            if hasattr(model, 'update_masks'):
                # We want to update_masks even for first ever batch.
                model.update_masks()

            model_out = model(xb)
            model_logits.append(model_out)
            if xbhat is None:
                xbhat = torch.zeros_like(model_out)
            xbhat += model_out

        if xbhat.shape == xb.shape:
            if mean:
                xb = (xb * std) + mean
            loss = F.binary_cross_entropy_with_logits(
                xbhat, xb, size_average=False) / xbhat.size()[0]
        else:
            if model.input_bins is None:
                # NOTE: we have to view() it in this order due to the mask
                # construction within MADE.  The masks there on the output unit
                # determine which unit sees what input vars.
                xbhat = xbhat.view(-1, model.nout // model.nin, model.nin)
                # Equivalent to:
                loss = F.cross_entropy(xbhat, xb.long(), reduction='none') \
                        .sum(-1).mean()
            else:
                if num_orders_to_forward == 1:
                    loss = model.nll(xbhat, xb).mean()
                else:
                    # Average across orderings & then across minibatch.
                    #
                    #   p(x) = 1/N sum_i p_i(x)
                    #   log(p(x)) = log(1/N) + log(sum_i p_i(x))
                    #             = log(1/N) + logsumexp ( log p_i(x) )
                    #             = log(1/N) + logsumexp ( - nll_i (x) )
                    #
                    # Used only at test time.
                    logps = []  # [batch size, num orders]
                    assert len(model_logits) == num_orders_to_forward, len(
                        model_logits)
                    for logits in model_logits:
                        # Note the minus.
                        logps.append(-model.nll(logits, xb))
                    logps = torch.stack(logps, dim=1)
                    logps = logps.logsumexp(dim=1) + torch.log(
                        torch.tensor(1.0 / nsamples, device=logps.device))
                    loss = (-logps).mean()

        losses.append(loss.item())

        if step % log_every == 0:
            if split == 'train':
                print(
                    'Epoch {} Iter {}, {} entropy gap {:.4f} bits (loss {:.3f}, data {:.3f}) {:.5f} lr'
                    .format(epoch_num, step, split,
                            loss.item() / np.log(2) - table_bits,
                            loss.item() / np.log(2), table_bits, opt.param_groups[0]['lr']))
            else:
                print('Epoch {} Iter {}, {} loss {:.4f} nats / {:.4f} bits'.
                      format(epoch_num, step, split, loss.item(),
                             loss.item() / np.log(2)))

        if split == 'train':
            opt.zero_grad()
            loss.backward()
            opt.step()

        if verbose:
            print('%s epoch average loss: %f' % (split, np.mean(losses)))
    if return_losses:
        return losses
    return np.mean(losses)

def RunFinetuneEpoch(split,
             model,
             opt,
             scheduler,
             train_data,
             val_data=None,
             batch_size=100,
             iterations=10,
             upto=None,
             epoch_num=None,
             verbose=False,
             direction='descent',
             log_every=10,
             return_losses=False,
             table_bits=None):
    torch.set_grad_enabled(split == 'train')
    model.train() if split == 'train' else model.eval()
    dataset = train_data if split == 'train' else val_data
    losses = []

    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=(split == 'train'))

    # How many orderings to run for the same batch?
    nsamples = 1
    if hasattr(model, 'orderings'):
        nsamples = len(model.orderings)

    if direction == "descent":
        for g in opt.param_groups:
            g['lr'] = g['lr']*10

    for step, xb in enumerate(loader):
        if split == 'train':
            if args.warmups:
                base_lr = 8e-4
                for param_group in opt.param_groups:
                    t = args.warmups
                    d_model = model.embed_size
                    global_steps = len(loader) * epoch_num + step + 1
                    lr = (d_model**-0.5) * min(
                        (global_steps**-.5), global_steps * (t**-1.5))

                    param_group['lr'] = lr

        if upto and step >= upto:
            break


        xb = xb.to(DEVICE).to(torch.float32)

        # Forward pass, potentially through several orderings.
        xbhat = None
        model_logits = []
        num_orders_to_forward = 1
        if split == 'test' and nsamples > 1:
            # At test, we want to test the 'true' nll under all orderings.
            num_orders_to_forward = nsamples

        for i in range(num_orders_to_forward):
            if hasattr(model, 'update_masks'):
                # We want to update_masks even for first ever batch.
                model.update_masks()

            model_out = model(xb)
            model_logits.append(model_out)
            if xbhat is None:
                xbhat = torch.zeros_like(model_out)
            xbhat += model_out

        if xbhat.shape == xb.shape:
            if mean:
                xb = (xb * std) + mean
            loss = F.binary_cross_entropy_with_logits(
                xbhat, xb, size_average=False) / xbhat.size()[0]
        else:
            if model.input_bins is None:
                # NOTE: we have to view() it in this order due to the mask
                # construction within MADE.  The masks there on the output unit
                # determine which unit sees what input vars.
                xbhat = xbhat.view(-1, model.nout // model.nin, model.nin)
                # Equivalent to:
                loss = F.cross_entropy(xbhat, xb.long(), reduction='none') \
                        .sum(-1).mean()
            else:
                if num_orders_to_forward == 1:
                    loss = model.nll(xbhat, xb).mean()
                else:
                    # Average across orderings & then across minibatch.
                    #
                    #   p(x) = 1/N sum_i p_i(x)
                    #   log(p(x)) = log(1/N) + log(sum_i p_i(x))
                    #             = log(1/N) + logsumexp ( log p_i(x) )
                    #             = log(1/N) + logsumexp ( - nll_i (x) )
                    #
                    # Used only at test time.
                    logps = []  # [batch size, num orders]
                    assert len(model_logits) == num_orders_to_forward, len(
                        model_logits)
                    for logits in model_logits:
                        # Note the minus.
                        logps.append(-model.nll(logits, xb))
                    logps = torch.stack(logps, dim=1)
                    logps = logps.logsumexp(dim=1) + torch.log(
                        torch.tensor(1.0 / nsamples, device=logps.device))
                    loss = (-logps).mean()

        losses.append(loss.item())

        if step % log_every == 0:
            if split == 'train':
                print(
                    'Epoch {} Iter {}, {} entropy gap {:.4f} bits (loss {:.3f}, data {:.3f}) {:.5f} lr'
                    .format(epoch_num, step, split,
                            loss.item() / np.log(2) - table_bits,
                            loss.item() / np.log(2), table_bits, opt.param_groups[0]['lr']))
            else:
                print('Epoch {} Iter {}, {} loss {:.4f} nats / {:.4f} bits'.
                      format(epoch_num, step, split, loss.item(),
                             loss.item() / np.log(2)))

        if split == 'train':
            if direction == "ascent":
                loss = -loss
            opt.zero_grad()
            loss.backward()
            opt.step()

        if verbose:
            print('%s epoch average loss: %f' % (split, np.mean(losses)))


    if direction == "descent":
        for g in opt.param_groups:
            g['lr'] = g['lr']/10

    if scheduler and direction == 'descent':
        scheduler.step()

    if return_losses:
        return losses
    return np.mean(losses)

def RunNegativeGradEpoch(split,
             model,
             opt,
             scheduler,
             train_data,
             deleted_data,
             val_data=None,
             batch_size=32,
             del_batch_size=32,
             iterations=10,
             upto=None,
             epoch_num=None,
             verbose=False,
             log_every=10,
             return_losses=False,
             table_bits=None):
    torch.set_grad_enabled(split == 'train')
    model.train() if split == 'train' else model.eval()
    dataset = train_data if split == 'train' else val_data
    losses = []


    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=(split == 'train'))

    dloader = torch.utils.data.DataLoader(deleted_data,
                                         batch_size=del_batch_size,
                                         shuffle=(split == 'train'))


    # How many orderings to run for the same batch?
    nsamples = 1
    if hasattr(model, 'orderings'):
        nsamples = len(model.orderings)
    step = -1
    for xb, delb in zip(loader, cycle(dloader)):
        step += 1

        xb = xb.to(DEVICE).to(torch.float32)
        delb = delb.to(DEVICE).to(torch.float32)

        # Forward pass, potentially through several orderings.
        xbhat = None
        delbhat = None
        model_logits = []
        model_del_logits = []
        num_orders_to_forward = 1
        if split == 'test' and nsamples > 1:
            # At test, we want to test the 'true' nll under all orderings.
            num_orders_to_forward = nsamples

        for i in range(num_orders_to_forward):
            if hasattr(model, 'update_masks'):
                # We want to update_masks even for first ever batch.
                model.update_masks()

            model_out = model(xb)
            model_del_out = model(delb)
            model_logits.append(model_out)
            model_del_logits.append(model_del_out)
            if xbhat is None:
                xbhat = torch.zeros_like(model_out)
            if delbhat is None:
                delbhat = torch.zeros_like(model_del_out)

            xbhat += model_out
            delbhat += model_del_out

        if xbhat.shape == xb.shape:
            if mean:
                xb = (xb * std) + mean
            loss1 = F.binary_cross_entropy_with_logits(
                xbhat, xb, size_average=False) / xbhat.size()[0]
            loss2 = F.binary_cross_entropy_with_logits(
                delbhat, delb, size_average=False) / delbhat.size()[0]
            loss = args.alpha*loss1 - (1-args.alpha)*loss2
        else:
            if model.input_bins is None:
                # NOTE: we have to view() it in this order due to the mask
                # construction within MADE.  The masks there on the output unit
                # determine which unit sees what input vars.
                xbhat = xbhat.view(-1, model.nout // model.nin, model.nin)
                delbhat = delbhat.view(-1, model.nout // model.nin, model.nin)
                # Equivalent to:
                loss1 = F.cross_entropy(xbhat, xb.long(), reduction='none') \
                        .sum(-1).mean()
                loss2 = F.cross_entropy(delbhat, delb.long(), reduction='none') \
                        .sum(-1).mean()
                loss = args.alpha*loss1 - (1-args.alpha)*loss2
            else:
                if num_orders_to_forward == 1:
                    loss1 = model.nll(xbhat, xb).mean()
                    loss2 = model.nll(delbhat, delb).mean()
                    loss = args.alpha*loss1 - (1-args.alpha)*loss2



        losses.append(loss.item())


        if step % log_every == 0:
            if split == 'train':
                print(
                    'Epoch {} Iter {}, {} entropy gap {:.4f} bits (loss {:.3f}, data {:.3f}) {:.5f} lr'
                    .format(epoch_num, step, split,
                            loss.item() / np.log(2) - table_bits,
                            loss.item() / np.log(2), table_bits, opt.param_groups[0]['lr']))
            else:
                print('Epoch {} Iter {}, {} loss {:.4f} nats / {:.4f} bits'.
                      format(epoch_num, step, split, loss.item(),
                             loss.item() / np.log(2)))

        if split == 'train':
            opt.zero_grad()
            loss.backward()
            opt.step()

        if verbose:
            print('%s epoch average loss: %f' % (split, np.mean(losses)))


    if return_losses:
        return losses
    return np.mean(losses)

def RunOneTeacherEpoch(split,
             model,
             teacher,
             opt,
             scheduler,
             transfer_data,
             batch_size=100,
             chance_loss=None,
             upto=None,
             epoch_num=None,
             verbose=False,
             log_every=10,
             return_losses=False,
             table_bits=None):
    torch.set_grad_enabled(True)
    model.train() 
    teacher.eval()
    dataset = transfer_data
    losses = []

    trloader = torch.utils.data.DataLoader(transfer_data,
                                         batch_size=batch_size,
                                         shuffle=True)

    # How many orderings to run for the same batch?
    nsamples = 1
    if hasattr(model, 'orderings'):
        nsamples = len(model.orderings)

    for step, trb in enumerate(trloader):

        if split == 'minimize':
            if args.warmups:
                base_lr = 8e-4
                for param_group in opt.param_groups:
                    t = args.warmups
                    d_model = model.embed_size
                    global_steps = len(trloader) * epoch_num + step + 1
                    lr = (d_model**-0.5) * min(
                        (global_steps**-.5), global_steps * (t**-1.5))

                    param_group['lr'] = lr


        if upto and step >= upto:
            break

        trb = trb.to(DEVICE).to(torch.float32)

        # Forward pass, potentially through several orderings.

        trbhat = None
        pmtrbhat = None
        model_logits = []
        model_tr_logits = []
        pmodel_logits = []
        num_orders_to_forward = 1
        if split == 'test' and nsamples > 1:
            # At test, we want to test the 'true' nll under all orderings.
            num_orders_to_forward = nsamples

        for i in range(num_orders_to_forward):
            if hasattr(model, 'update_masks'):
                # We want to update_masks even for first ever batch.
                model.update_masks()


            model_tr_out = model(trb)
            pmodel_out = teacher(trb)

            model_tr_logits.append(model_tr_out)
            pmodel_logits.append(pmodel_out)
            if trbhat is None:

                trbhat = torch.zeros_like(model_tr_out)
                pmtrbhat = torch.zeros_like(pmodel_out)
 
            trbhat += model_tr_out
            pmtrbhat += pmodel_out

        if trbhat.shape == trb.shape:
            if mean:
                trb = (trb * std) + mean

            reg_loss = F.binary_cross_entropy_with_logits(
                trbhat, trb, size_average=False) / trbhat.size()[0]

            kd_loss = model.kd_loss(trbhat, pmtrbhat).mean()



        if num_orders_to_forward == 1:

            nll2 = model.nll(trbhat, trb)
            #nll3 = pmodel.nll(pmtrbhat, trb)

            reg_loss = nll2.mean()
            kd_loss = model.kd_loss(trbhat, pmtrbhat).mean()
            
 

        if split == "minimize":
            loss = args.alpha*kd_loss + (1-args.alpha)*reg_loss
        if split == "maximize":
            if chance_loss is None:
                loss = -kd_loss
            else:
                loss = -torch.clamp(kd_loss, max=10*chance_loss)
        if split == "test":
            loss = kd_loss


        if split != "test":
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        if split == "maximize":
            loss = -loss
        losses.append(loss.item())

        if step % log_every == 0 and split != "test":

            print(
                'Epoch {} Iter {}, {} entropy gap {:.4f} bits (loss {:.3f}, data {:.3f}) {:.5f} lr'
                .format(epoch_num, step, split,
                        loss.item() / np.log(2) - table_bits,
                        loss.item() / np.log(2), table_bits, opt.param_groups[0]['lr']))



        if verbose:
            print('%s epoch average loss: %f' % (split, np.mean(losses)))

    if scheduler:
        scheduler.step()

    if return_losses:
        return losses
    return np.mean(losses)

def RunTwoTeacherEpoch(split,
             model,
             gteacher,
             bteacher,
             opt,
             scheduler,
             transfer_data,
             deleted_data,
             alpha,
             val_data=None,
             batch_size=100,
             upto=None,
             epoch_num=None,
             verbose=False,
             log_every=10,
             return_losses=False,
             table_bits=None):
    torch.set_grad_enabled(split == 'train')
    model.train() if split == 'train' else model.eval()
    gteacher.eval()
    bteacher.eval()
    dataset = transfer_data if split == 'train' else val_data
    losses = []

    trloader = torch.utils.data.DataLoader(transfer_data,
                                         batch_size=batch_size,
                                         shuffle=(split == 'train'))

    dloader = torch.utils.data.DataLoader(deleted_data,
                                         batch_size=batch_size,
                                         shuffle=(split == 'train'))

    # How many orderings to run for the same batch?
    nsamples = 1
    if hasattr(model, 'orderings'):
        nsamples = len(model.orderings)

    for step, trb in enumerate(trloader):
        delb = next(iter(dloader))
        if split == 'train':
            if args.warmups:
                base_lr = 8e-4
                for param_group in opt.param_groups:
                    t = args.warmups
                    d_model = model.embed_size
                    global_steps = len(trloader) * epoch_num + step + 1
                    lr = (d_model**-0.5) * min(
                        (global_steps**-.5), global_steps * (t**-1.5))

                    param_group['lr'] = lr


        if upto and step >= upto:
            break

        trb = trb.to(DEVICE).to(torch.float32)
        delb = delb.to(DEVICE).to(torch.float32)

        # Forward pass, potentially through several orderings.

        trbhat = None
        delbhat = None
        gmtrbhat = None
        bmtrbhat = None
        model_logits = []
        model_tr_logits = []
        model_del_logits = []
        gmodel_logits = []
        bmodel_logits = []
        num_orders_to_forward = 1
        if split == 'test' and nsamples > 1:
            # At test, we want to test the 'true' nll under all orderings.
            num_orders_to_forward = nsamples

        for i in range(num_orders_to_forward):
            if hasattr(model, 'update_masks'):
                # We want to update_masks even for first ever batch.
                model.update_masks()


            model_tr_out = model(trb)
            model_del_out = model(delb)
            gmodel_out = gteacher(trb)
            bmodel_out = bteacher(delb)

            model_tr_logits.append(model_tr_out)
            model_del_logits.append(model_del_out)
            gmodel_logits.append(gmodel_out)
            bmodel_logits.append(bmodel_out)

            if trbhat is None:

                trbhat = torch.zeros_like(model_tr_out)
                delbhat = torch.zeros_like(model_del_out)
                gmtrbhat = torch.zeros_like(gmodel_out)
                bmtrbhat = torch.zeros_like(bmodel_out)
 
            trbhat += model_tr_out
            delbhat += model_del_out
            gmtrbhat += gmodel_out
            bmtrbhat += bmodel_out

        if trbhat.shape == trb.shape:
            if mean:
                trb = (trb * std) + mean
                delb = (delb * std) + mean

            #reg_loss = F.binary_cross_entropy_with_logits(
            #    trbhat, trb, size_average=False) / trbhat.size()[0]

            kd_loss_good = KD_loss(trbhat, gmtrbhat).mean()
            kd_loss_bad = KD_loss(delbhat, bmtrbhat).mean()

            loss = alpha*(kd_loss_good) - (1-alpha)*kd_loss_bad



        if num_orders_to_forward == 1:

            nll2 = model.nll(trbhat, trb)
            #nll3 = pmodel.nll(pmtrbhat, trb)

            #reg_loss = nll2.mean()
            kd_loss_good = model.kd_loss(trbhat, gmtrbhat).mean()
            kd_loss_bad = model.kd_loss(delbhat, bmtrbhat).mean()

            loss = alpha*(kd_loss_good) - (1-alpha)*kd_loss_bad
 

        
        losses.append(loss.item())

        if step % log_every == 0:
            if split == 'train':
                print(
                    'Epoch {} Iter {}, {} entropy gap {:.4f} bits (loss {:.3f}, data {:.3f}) {:.5f} lr'
                    .format(epoch_num, step, split,
                            loss.item() / np.log(2) - table_bits,
                            loss.item() / np.log(2), table_bits, opt.param_groups[0]['lr']))
            else:
                print('Epoch {} Iter {}, {} loss {:.4f} nats / {:.4f} bits'.
                      format(epoch_num, step, split, loss.item(),
                             loss.item() / np.log(2)))

        if split == 'train':
            opt.zero_grad()
            loss.backward()
            opt.step()

        if verbose:
            print('%s epoch average loss: %f' % (split, np.mean(losses)))

    if scheduler:
        scheduler.step()

    if return_losses:
        return losses
    return np.mean(losses)

def RunTwoTeacherMinimizeEpoch(split,
             model,
             teacher,
             opt,
             scheduler,
             transfer_data,
             deleted_data,
             ood_data,
             alpha,
             val_data=None,
             batch_size=100,
             upto=None,
             epoch_num=None,
             verbose=False,
             log_every=10,
             return_losses=False,
             table_bits=None):
    torch.set_grad_enabled(split == 'train')
    model.train() if split == 'train' else model.eval()
    teacher.eval()
    dataset = transfer_data if split == 'train' else val_data
    losses = []

    trloader = torch.utils.data.DataLoader(transfer_data,
                                         batch_size=batch_size,
                                         shuffle=(split == 'train'))

    dloader = torch.utils.data.DataLoader(deleted_data,
                                         batch_size=batch_size,
                                         shuffle=False)
    oloader = torch.utils.data.DataLoader(ood_data,
                                         batch_size=batch_size,
                                         shuffle=False)

    # How many orderings to run for the same batch?
    nsamples = 1
    if hasattr(model, 'orderings'):
        nsamples = len(model.orderings)

    for step, trb in enumerate(trloader):
        delb = next(iter(dloader))
        oodb = next(iter(oloader))
        if split == 'train':
            if args.warmups:
                base_lr = 8e-4
                for param_group in opt.param_groups:
                    t = args.warmups
                    d_model = model.embed_size
                    global_steps = len(trloader) * epoch_num + step + 1
                    lr = (d_model**-0.5) * min(
                        (global_steps**-.5), global_steps * (t**-1.5))

                    param_group['lr'] = lr


        if upto and step >= upto:
            break

        trb = trb.to(DEVICE).to(torch.float32)
        delb = delb.to(DEVICE).to(torch.float32)
        oodb = oodb.to(DEVICE).to(torch.float32)

        # Forward pass, potentially through several orderings.

        trbhat = None
        delbhat = None
        oodbhat = None

        gmtrbhat = None
        bmtrbhat = None

        model_logits = []
        model_tr_logits = []
        model_del_logits = []
        gmodel_logits = []
        bmodel_logits = []
        num_orders_to_forward = 1
        if split == 'test' and nsamples > 1:
            # At test, we want to test the 'true' nll under all orderings.
            num_orders_to_forward = nsamples

        for i in range(num_orders_to_forward):
            if hasattr(model, 'update_masks'):
                # We want to update_masks even for first ever batch.
                model.update_masks()


            model_tr_out = model(trb)
            model_del_out = model(delb)

            gmodel_out = teacher(trb)
            bmodel_out = teacher(oodb)

            model_tr_logits.append(model_tr_out)
            model_del_logits.append(model_del_out)
            gmodel_logits.append(gmodel_out)
            bmodel_logits.append(bmodel_out)

            if trbhat is None:

                trbhat = torch.zeros_like(model_tr_out)
                delbhat = torch.zeros_like(model_del_out)
                gmtrbhat = torch.zeros_like(gmodel_out)
                bmtrbhat = torch.zeros_like(bmodel_out)
 
            trbhat += model_tr_out
            delbhat += model_del_out
            gmtrbhat += gmodel_out
            bmtrbhat += bmodel_out

        if trbhat.shape == trb.shape:
            if mean:
                trb = (trb * std) + mean
                delb = (delb * std) + mean
                oodb = (oodb * std) + mean

            #reg_loss = F.binary_cross_entropy_with_logits(
            #    trbhat, trb, size_average=False) / trbhat.size()[0]

            kd_loss_good = KD_loss(trbhat, gmtrbhat).mean()
            kd_loss_bad = KD_loss(delbhat, bmtrbhat).mean()

            loss = alpha*(kd_loss_good) + (1-alpha)*kd_loss_bad



        if num_orders_to_forward == 1:

            #nll2 = model.nll(trbhat, trb)
            #nll3 = pmodel.nll(pmtrbhat, trb)

            #reg_loss = nll2.mean()
            kd_loss_good = model.kd_loss(trbhat, gmtrbhat).mean()
            kd_loss_bad = model.kd_loss(delbhat, bmtrbhat).mean()

            loss = alpha*(kd_loss_good) + (1-alpha)*kd_loss_bad
 

        
        losses.append(loss.item())

        if step % log_every == 0:
            if split == 'train':
                print(
                    'Epoch {} Iter {}, {} entropy gap {:.4f} bits (loss {:.3f}, data {:.3f}) {:.5f} lr'
                    .format(epoch_num, step, split,
                            loss.item() / np.log(2) - table_bits,
                            loss.item() / np.log(2), table_bits, opt.param_groups[0]['lr']))
            else:
                print('Epoch {} Iter {}, {} loss {:.4f} nats / {:.4f} bits'.
                      format(epoch_num, step, split, loss.item(),
                             loss.item() / np.log(2)))

        if split == 'train':
            opt.zero_grad()
            loss.backward()
            opt.step()

        if verbose:
            print('%s epoch average loss: %f' % (split, np.mean(losses)))

    if scheduler:
        scheduler.step()

    if return_losses:
        return losses
    return np.mean(losses)

def ReportModel(model, blacklist=None):
    ps = []
    for name, p in model.named_parameters():
        if blacklist is None or blacklist not in name:
            ps.append(np.prod(p.size()))
    num_params = sum(ps)
    mb = num_params * 4 / 1024 / 1024
    print('Number of model parameters: {} (~= {:.1f}MB)'.format(num_params, mb))
    print(model)
    return mb

def InvertOrder(order):
    if order is None:
        return None
    # 'order'[i] maps nat_i -> position of nat_i
    # Inverse: position -> natural idx.  This it the 'true' ordering -- it's how
    # heuristic orders are generated + (less crucially) how Transformer works.
    nin = len(order)
    inv_ordering = [None] * nin
    for natural_idx in range(nin):
        inv_ordering[order[natural_idx]] = natural_idx
    return inv_ordering

def MakeMade(scale, cols_to_train, seed, fixed_ordering=None):
    if args.inv_order:
        print('Inverting order!')
        fixed_ordering = InvertOrder(fixed_ordering)

    model = made.MADE(
        nin=len(cols_to_train),
        hidden_sizes=[scale] *
        args.layers if args.layers > 0 else [512, 256, 512, 128, 1024],
        nout=sum([c.DistributionSize() for c in cols_to_train]),
        input_bins=[c.DistributionSize() for c in cols_to_train],
        input_encoding=args.input_encoding,
        output_encoding=args.output_encoding,
        embed_size=32,
        seed=seed,
        do_direct_io_connections=args.direct_io,
        natural_ordering=False if seed is not None and seed != 0 else True,
        residual_connections=args.residual,
        fixed_ordering=fixed_ordering,
        column_masking=args.column_masking,
    ).to(DEVICE)

    return model

def MakeTransformer(cols_to_train, fixed_ordering, seed=None):
    return transformer.Transformer(
        num_blocks=args.blocks,
        d_model=args.dmodel,
        d_ff=args.dff,
        num_heads=args.heads,
        nin=len(cols_to_train),
        input_bins=[c.DistributionSize() for c in cols_to_train],
        use_positional_embs=True,
        activation=args.transformer_act,
        fixed_ordering=fixed_ordering,
        column_masking=args.column_masking,
        seed=seed,
    ).to(DEVICE)

def InitWeight(m):
    if type(m) == made.MaskedLinear or type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    if type(m) == nn.Embedding:
        nn.init.normal_(m.weight, std=0.02)

def TrainTask(args, seed=0, logger=None):
    torch.manual_seed(0)
    np.random.seed(0)

    assert args.dataset in DATASETS
    if args.dataset == 'census':
        table = datasets.load_census()
    elif args.dataset == 'forest':
        table = datasets.load_forest()
    elif args.dataset == 'dmv':
        table = datasets.load_DMV()
    table_bits = Entropy(
        table,
        table.data.fillna(value=0).groupby([c.name for c in table.columns
                                           ]).size(), [2])[0]
    fixed_ordering = None

    if args.order is not None:
        print('Using passed-in order:', args.order)
        fixed_ordering = args.order

    print(table.data.info())

    table_train = table

    if args.heads > 0:
        model = MakeTransformer(cols_to_train=table.columns,
                                fixed_ordering=fixed_ordering,
                                seed=seed)
        if args.pre_model:
            model.load_state_dict(torch.load(args.pre_model))
    else:
        if args.dataset in DATASETS:
            model = MakeMade(
                scale=args.fc_hiddens,
                cols_to_train=table.columns,
                seed=seed,
                fixed_ordering=fixed_ordering,
            )
            if args.pre_model:
                model.load_state_dict(torch.load(args.pre_model))
        else:
            assert False, args.dataset

    mb = ReportModel(model)

    if not isinstance(model, transformer.Transformer):
        print('Applying InitWeight()')
        model.apply(InitWeight)

    if isinstance(model, transformer.Transformer):
        lr = args.learning_rate
        opt = torch.optim.Adam(
            list(model.parameters()),
            lr,
            betas=(0.9, 0.98),
            eps=1e-9,
        )
    else:
        lr = args.learning_rate
        opt = torch.optim.Adam(list(model.parameters()), lr)

    bs = args.bs
    log_every = 200

    train_data = common.TableDataset(table_train)


    if isinstance(logger, type(wandb.run)) and logger is not None:
        logger.watch(model)
        logger.name = args.dataset+"_"+args.mode


    train_losses = []
    train_start = time.time()
    for epoch in range(1, args.epochs+1):

        mean_epoch_train_loss = RunEpoch('train',
                                         model,
                                         opt,
                                         train_data=train_data,
                                         val_data=train_data,
                                         batch_size=bs,
                                         epoch_num=epoch,
                                         log_every=log_every,
                                         table_bits=table_bits)


        if epoch % 1 == 0:
            print('epoch {} train loss {:.4f} nats / {:.4f} bits'.format(
                epoch, mean_epoch_train_loss,
                mean_epoch_train_loss / np.log(2)))
            since_start = time.time() - train_start
            print('time since start: {:.1f} secs'.format(since_start))


        model_nats = np.mean(mean_epoch_train_loss)
        model_bits = model_nats / np.log(2)
        model.model_bits = model_bits
        PATH = 'models/{}{}-{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}-{}epochs-seed{}.pt'.format(
            epoch, args.dataset, args.mode, mb, model.model_bits, table_bits, model.name(),
            args.epochs, seed)
        
        #uncomment if needed intermediate evaluations
        #os.makedirs(os.path.dirname(PATH), exist_ok=True)
        #torch.save(model.state_dict(), PATH)
 
        train_losses.append(mean_epoch_train_loss)

        eval(model, mean_epoch_train_loss, 0, PATH, args.dataset, epoch, args, logger)

    train_stop = time.time()
    print('Training done; evaluating likelihood on full data:')
    all_losses = RunEpoch('test',
                          model,
                          train_data=train_data,
                          val_data=train_data,
                          opt=None,
                          batch_size=1024,
                          log_every=500,
                          table_bits=table_bits,
                          return_losses=True)
    model_nats = np.mean(all_losses)
    model_bits = model_nats / np.log(2)
    model.model_bits = model_bits

    if fixed_ordering is None:
        if seed is not None:
            PATH = 'models/{}-{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}-{}epochs-seed{}.pt'.format(
                args.dataset, args.mode, mb, model.model_bits, table_bits, model.name(),
                args.epochs, seed)
        else:
            PATH = 'models/{}-{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}-{}epochs-seed{}-{}.pt'.format(
                args.dataset, args.mode, mb, model.model_bits, table_bits, model.name(),
                args.epochs, seed, time.time())
    else:
        annot = ''
        if args.inv_order:
            annot = '-invOrder'

        PATH = 'models/{}-{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}-{}epochs-seed{}-order{}{}.pt'.format(
            args.dataset, args.mode, mb, model.model_bits, table_bits, model.name(),
            args.epochs, seed, '_'.join(map(str, fixed_ordering)), annot)

    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    torch.save(model.state_dict(), PATH)
    print('Saved to:')
    print(PATH)

    eval(model, model_nats, (train_stop - train_start)/60, PATH, args.dataset, -1, args, logger)

def UnlearnTask(args, seed=0, logger=None):

    assert os.path.exists(args.filters)

    torch.manual_seed(0)
    np.random.seed(0)

    assert args.dataset in DATASETS
    if args.dataset == 'census':
        #filters = {'range':('age', 0, 30)}     #conditions to remove data
        original, retained, deleted = datasets.load_reduced_census(args.filters, frac=1)
    if args.dataset == 'forest':
        original, retained, deleted = datasets.load_reduced_forest(args.filters, frac=1)
        #permuted = datasets.load_permuted_forest(filename='forest_reduced.csv', size=len(deleted.data))
    if args.dataset == 'dmv':
        original, retained, deleted = datasets.load_reduced_DMV(args.filters, frac=1)

    table_bits = Entropy(
        retained,
        retained.data.fillna(value=0).groupby([c.name for c in retained.columns
                                           ]).size(), [2])[0]
    fixed_ordering = None

    if args.order is not None:
        print('Using passed-in order:', args.order)
        fixed_ordering = args.order

    print(retained.data.info())


    if args.heads > 0:
        model = MakeTransformer(cols_to_train=original.columns,
                                fixed_ordering=fixed_ordering,
                                seed=seed)
        if args.pre_model:
            model.load_state_dict(torch.load(args.pre_model))
    else:
        if args.dataset in DATASETS:
            model = MakeMade(
                scale=args.fc_hiddens,
                cols_to_train=original.columns,
                seed=seed,
                fixed_ordering=fixed_ordering,
            )
            if args.pre_model:
                model.load_state_dict(torch.load(args.pre_model))
        else:
            assert False, args.dataset

    teacher = deepcopy(model)
    teacher.eval()

    mb = ReportModel(model)

    if args.remove_weights:
        print('Applying InitWeight()')
        model.apply(InitWeight)

    if isinstance(model, transformer.Transformer):
        lr = args.learning_rate
        opt = torch.optim.Adam(
            list(model.parameters()),
            lr,
            betas=(0.9, 0.98),
            eps=1e-9,
        )
    else:
        lr = args.learning_rate
        opt = torch.optim.Adam(list(model.parameters()), lr)

    lr_scheduler = None
    if args.warmups == 0:
        lr = args.learning_rate
        opt = torch.optim.Adam(list(model.parameters()), lr=lr)
        decay_rate = 0.98
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=opt, gamma=decay_rate
        )
    bs = args.bs
    del_bs = int(bs)
    log_every = 200


    train_data = common.TableDataset(retained, original)
    deleted_set = common.TableDataset(deleted, original)
    #permute_set = common.TableDataset(permuted, original)
    #permute_set.tuples = permute_set.tuples.repeat(len(deleted_set),1)
    sample = deepcopy(train_data)


    frac = args.tr_frac
    num_samples = int(len(train_data)*frac)
    rndindices = torch.randperm(len(train_data))[:num_samples]
    sample.tuples = train_data.tuples[rndindices]

    print (train_data.tuples.shape)

    if isinstance(logger, type(wandb.run)) and logger is not None:
        logger.config.update({"deleted size": len(deleted_set)})
        logger.name = args.dataset+"-"+args.mode+"_lendel{}".format(len(deleted_set))
        logger.watch(model)


    scratch_model = deepcopy(model)
    scratch_model.apply(InitWeight)

    all_losses = RunOneTeacherEpoch('test',
                                     model,
                                     scratch_model,
                                     opt,
                                     scheduler = lr_scheduler,
                                     transfer_data=deleted_set,
                                     batch_size=bs,
                                     epoch_num=0,
                                     log_every=log_every,
                                     table_bits=table_bits)
    chance_loss = np.mean(all_losses)
    divergence_loss = 0
    print ("chance loss for the deleted set:", chance_loss)
    train_losses = []
    train_start = time.time()

    if args.mode == "stale":
        assert args.epochs == 0
    for epoch in range(1, args.epochs+1):

        if args.mode == 'retrain':
            mean_epoch_train_loss = RunEpoch('train',
                                             model,
                                             opt,
                                             train_data=train_data,
                                             val_data=train_data,
                                             batch_size=bs,
                                             epoch_num=epoch,
                                             log_every=log_every,
                                             table_bits=table_bits)
        elif args.mode == 'finetune':
            mean_epoch_train_loss = RunFinetuneEpoch('train',
                                             model,
                                             opt,
                                             scheduler=lr_scheduler,
                                             train_data=sample,
                                             val_data=sample,
                                             batch_size=bs,
                                             iterations=100,
                                             epoch_num=epoch,
                                             log_every=log_every,
                                             table_bits=table_bits)
        elif args.mode in ['negatgrad', 'negatgrad+']:
            mean_epoch_train_loss = RunNegativeGradEpoch('train',
                                             model,
                                             opt,
                                             scheduler=lr_scheduler,
                                             train_data=sample,
                                             val_data=sample,
                                             deleted_data = deleted_set,
                                             batch_size=bs,
                                             del_batch_size=args.del_bs,
                                             iterations=100,
                                             epoch_num=epoch,
                                             log_every=log_every,
                                             table_bits=table_bits)
        elif args.mode == 'sgda':
            if epoch % 20 == 0: 
                print ("ascending")
                mean_epoch_train_loss = RunFinetuneEpoch('train',
                                                 model,
                                                 opt,
                                                 scheduler=lr_scheduler,
                                                 train_data=deleted_set,
                                                 val_data=deleted_set,
                                                 batch_size=args.del_bs,
                                                 direction='ascent',
                                                 iterations=100,
                                                 epoch_num=epoch,
                                                 log_every=log_every,
                                                 table_bits=table_bits)
            mean_epoch_train_loss = RunFinetuneEpoch('train',
                                             model,
                                             opt,
                                             scheduler=lr_scheduler,
                                             train_data=sample,
                                             val_data=sample,
                                             batch_size=bs,
                                             direction='descent',
                                             iterations=100,
                                             epoch_num=epoch,
                                             log_every=log_every,
                                             table_bits=table_bits)
        elif args.mode == 'unlearn-one-teacher':
            mean_epoch_train_loss = RunOneTeacherEpoch('minimize',
                                             model,
                                             teacher,
                                             opt,
                                             scheduler = lr_scheduler,
                                             transfer_data=sample,
                                             batch_size=bs,
                                             epoch_num=epoch,
                                             alpha=0.5,
                                             log_every=log_every,
                                             table_bits=table_bits)

        elif args.mode == 'scrub':
            if epoch < args.msteps:
                mean_epoch_train_loss = RunOneTeacherEpoch('maximize',
                                                 model,
                                                 teacher,
                                                 opt,
                                                 scheduler = None,
                                                 transfer_data=deleted_set,
                                                 batch_size=del_bs,
                                                 epoch_num=epoch,
                                                 chance_loss=None,
                                                 log_every=log_every,
                                                 table_bits=table_bits)

                divergence_loss = mean_epoch_train_loss
            mean_epoch_train_loss = RunOneTeacherEpoch('minimize',
                                             model,
                                             teacher,
                                             opt,
                                             scheduler = lr_scheduler,
                                             transfer_data=sample,
                                             batch_size=bs,
                                             epoch_num=epoch,
                                             log_every=log_every,
                                             table_bits=table_bits)

        elif args.mode == "unlearn-two-teacher-linear":
            alpha = 0.9999
            if epoch >= 10:
                alpha = 1
            mean_epoch_train_loss = RunTwoTeacherEpoch('train',
                                             model,
                                             teacher,
                                             teacher,
                                             opt,
                                             scheduler = lr_scheduler,
                                             transfer_data=sample,
                                             val_data=sample,
                                             deleted_data=deleted_set,
                                             batch_size=bs,
                                             epoch_num=epoch,
                                             alpha=alpha,
                                             log_every=log_every,
                                             table_bits=table_bits)

        elif args.mode == "unlearn-two-teacher-bothminimize":
            if epoch <= 10:
                alpha = 0.1
                mean_epoch_train_loss = RunTwoTeacherMinimizeEpoch('train',
                                                 model,
                                                 teacher,
                                                 opt,
                                                 scheduler = lr_scheduler,
                                                 transfer_data=sample,
                                                 deleted_data=deleted_set,
                                                 ood_data=permute_set,
                                                 batch_size=bs,
                                                 epoch_num=epoch,
                                                 alpha=alpha,
                                                 log_every=log_every,
                                                 table_bits=table_bits)
            else:
                mean_epoch_train_loss = RunFinetuneEpoch('train',
                                                 model,
                                                 opt,
                                                 scheduler=lr_scheduler,
                                                 train_data=sample,
                                                 val_data=sample,
                                                 batch_size=bs,
                                                 iterations=100,
                                                 epoch_num=epoch,
                                                 log_every=log_every,
                                                 table_bits=table_bits)

        else:
            raise ValueError("unknown mode!")

        if epoch % 1 == 0:
            print('epoch {} train loss {:.4f} nats / {:.4f} bits'.format(
                epoch, mean_epoch_train_loss,
                mean_epoch_train_loss / np.log(2)))
            since_start = time.time() - train_start
            print('time since start: {:.1f} secs'.format(since_start))
        train_losses.append(mean_epoch_train_loss)

        model_nats = np.mean(mean_epoch_train_loss)
        model_bits = model_nats / np.log(2)
        model.model_bits = model_bits
        PATH = 'models/{}{}-{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}-{}epochs-seed{}.pt'.format(
            epoch, args.dataset, args.mode, mb, model.model_bits, table_bits, model.name(),
            args.epochs, seed)
        #uncommented if wanted intermediate evaluations
        #os.makedirs(os.path.dirname(PATH), exist_ok=True)
        #torch.save(model.state_dict(), PATH)

        eval(model, mean_epoch_train_loss, 0, PATH, args.dataset + "_reduced", epoch, args, logger)

    train_stop = time.time()

    print('Training done; evaluating likelihood on full data:')
    all_losses = RunEpoch('test',
                          model,
                          train_data=train_data,
                          val_data=train_data,
                          opt=None,
                          batch_size=1024,
                          log_every=500,
                          table_bits=table_bits,
                          return_losses=True)
    model_nats = np.mean(all_losses)
    model_bits = model_nats / np.log(2)
    model.model_bits = model_bits

    if fixed_ordering is None:
        if seed is not None:
            PATH = 'models/{}-{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}-{}epochs-seed{}.pt'.format(
                args.dataset, args.mode, mb, model.model_bits, table_bits, model.name(),
                args.epochs, seed)
        else:
            PATH = 'models/{}-{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}-{}epochs-seed{}-{}.pt'.format(
                args.dataset, args.mode, mb, model.model_bits, table_bits, model.name(),
                args.epochs, seed, time.time())
    else:
        annot = ''
        if args.inv_order:
            annot = '-invOrder'

        PATH = 'models/{}-{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}-{}epochs-seed{}-order{}{}.pt'.format(
            args.dataset, args.mode, mb, model.model_bits, table_bits, model.name(),
            args.epochs, seed, '_'.join(map(str, fixed_ordering)), annot)
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    torch.save(model.state_dict(), PATH)
    print('Saved to:')
    print(PATH)
    
    eval(model, model_nats, (train_stop - train_start)/60, PATH, args.dataset + "_reduced", -1, args, logger)

def eval(model, avg_loss, train_time, PATH, dataset, epoch, args, logger=None):

    if epoch != -1:
        if epoch % args.eval_per_epoch == 0:

            subprocess.call(["python", "eval_model.py", "--dataset={}".format(dataset), "--metric=relative",
                "--layers=5", "--fc-hiddens=256", "--direct-io","--column-masking", "--residual",
                "--num-queries=2000", "--glob={}".format(PATH)],
                stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)

            summary_results = pd.read_csv('summary_results.csv')
            for i in range(len(summary_results)):
                if summary_results.iloc[i]['est'].startswith('psample'):
                    mean = summary_results.iloc[i]['mean']
                    median = summary_results.iloc[i]['median']
                    p_90th = summary_results.iloc[i]['percentile_90']
                    p_95th = summary_results.iloc[i]['percentile_95']
                    p_99th = summary_results.iloc[i]['percentile_99']

            logger.log({
                "loss": avg_loss,
                "mean": mean, 
                "median": median,
                "90th-percentile": p_90th, "95th-percentile": p_95th,
                 "99th-percentile": p_99th}
                 )


            if args.mode != train:
                subprocess.call(["python", "eval_model.py", "--dataset={}".format(dataset), "--layers=5",
                    "--fc-hiddens=256", "--direct-io","--column-masking", "--residual",
                    "--mode=likelihood", "--glob={}".format(PATH)],
                    stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
                summary_likelihoods = pd.read_csv('likelihood_summary.csv')
                retained_loss = summary_likelihoods.iloc[0]['retain set']
                deleted_loss = summary_likelihoods.iloc[0]['deleted set']
                logger.log({
                     "retain-likelihood": retained_loss,
                     "deleted-likelihood": deleted_loss
                     })

            """
            os.remove('summary_results.csv')
            os.remove('previous_queries.pkl')
            subprocess.call(["python", "eval_model.py", "--dataset=census_reduced", "--metric=absolute",
                "--mode=eval-deleted", "--filter=age,0,30",
                "--layers=0", "--direct-io","--column-masking", "--input-encoding=binary",
                "--output-encoding=one_hot", "--num-queries=20", "--glob={}".format(PATH)],
                stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)


            summary_results = pd.read_csv('summary_results.csv')
            for i in range(len(summary_results)):
                if summary_results.iloc[i]['est'].startswith('psample'):
                    mean = summary_results.iloc[i]['mean']
                    median = summary_results.iloc[i]['median']
                    p_90th = summary_results.iloc[i]['percentile_90']
                    p_95th = summary_results.iloc[i]['percentile_95']
                    p_99th = summary_results.iloc[i]['percentile_99']
            

            wandb.log({
                 "mean-deleted": mean, 
                 "median-deleted": median,
                 "90th-percentile-deleted": p_90th,
                 "95th-percentile-deleted": p_95th,
                 "99th-percentile-deleted": p_99th}
                 )
            """

        else:
            logger.log({"loss": avg_loss})  


    elif epoch == -1:
        print (" ".join(["python", "eval_model.py", "--dataset={}".format(dataset), "--metric=relative",
                    "--layers=5", "--fc-hiddens=256", "--direct-io","--column-masking", "--residual",
                    "--num-queries=2000", "--glob={}".format(PATH)]))

        subprocess.call(["python", "eval_model.py", "--dataset={}".format(dataset), "--metric=relative",
                    "--layers=5", "--fc-hiddens=256", "--direct-io","--column-masking", "--residual",
                    "--num-queries=2000", "--glob={}".format(PATH)],
            stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)

        summary_results = pd.read_csv('summary_results.csv')
        for i in range(len(summary_results)):
            if summary_results.iloc[i]['est'].startswith('psample'):
                mean = summary_results.iloc[i]['mean']
                median = summary_results.iloc[i]['median']
                p_90th = summary_results.iloc[i]['percentile_90']
                p_95th = summary_results.iloc[i]['percentile_95']
                p_99th = summary_results.iloc[i]['percentile_99']

        logger.log({
             "final-mean": mean,
             "final-median": median,
             "final-90th-percentile": p_90th,
             "final-95th-percentile": p_95th,
             "final-99th-percentile": p_99th,
             "train time": train_time,
             })


        """
        if args.mode != 'train':
            subprocess.call(["python", "eval_model.py", "--dataset={}".format(dataset), "--mode=likelihood",
                        "--layers=5", "--fc-hiddens=256", "--direct-io","--column-masking", "--residual",
                        "--glob={}".format(PATH)],
                stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)

            summary_likelihoods = pd.read_csv('likelihood_summary.csv')
            retained_loss = summary_likelihoods.iloc[0]['retain set']
            deleted_loss = summary_likelihoods.iloc[0]['deleted set']
            wandb.log({
                 "final-retain-likelihood": retained_loss,
                 "final-deleted-likelihood": deleted_loss
                 })

        """
        
        if args.eval_deleted:
            os.remove('summary_results.csv')
            os.remove('previous_queries.pkl')

            subprocess.call(["python", "eval_model.py", "--dataset={}".format(dataset), "--metric=absolute",
                "--mode=eval-deleted", "--filter=age,0,30",
                "--layers=5", "--fc-hiddens=256", "--direct-io","--column-masking", "--residual",
                "--num-queries=2000", "--glob={}".format(PATH)],
                stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)

            summary_results = pd.read_csv('summary_results.csv')
            for i in range(len(summary_results)):
                if summary_results.iloc[i]['est'].startswith('psample'):
                    mean = summary_results.iloc[i]['mean']
                    median = summary_results.iloc[i]['median']
                    p_90th = summary_results.iloc[i]['percentile_90']
                    p_95th = summary_results.iloc[i]['percentile_95']
                    p_99th = summary_results.iloc[i]['percentile_99']

            logger.log({
                 "final-deleted-mean": mean,
                 "final-deleted-median": median,
                 "final-deleted-90th-percentile": p_90th,
                 "final-deleted-95th-percentile": p_95th,
                 "final-deleted-99th-percentile": p_99th,
                 })
            
            os.remove('summary_results.csv')
            os.remove('previous_queries.pkl')


        if isinstance(logger, type(wandb.run)) and logger is not None:
            """
            artifact = wandb.Artifact('model', type='model')
            artifact.add_file(PATH)
            logger.log_artifact(artifact)

            artifact = wandb.Artifact('filters', type='file')
            artifact.add_file(args.filters)
            logger.log_artifact(artifact)

            artifact = wandb.Artifact('queries', type='file')
            artifact.add_file("previous_queries.pkl")
            logger.log_artifact(artifact)

            artifact = wandb.Artifact('results', type='file')
            artifact.add_file("results.csv")
            logger.log_artifact(artifact)
            """
            logger.save(PATH)
            logger.save(args.filters)
            shutil.copyfile('previous_queries.pkl', 'previous_queries_tosave.pkl') #a trick because of wandb's asyncrounous save
            shutil.copyfile('results.csv', 'results_tosave.csv') 
            logger.save("previous_queries_tosave.pkl")
            logger.save("results_tosave.csv")


        for file in ['results.csv', 'summary_results.csv', 'likelihood_summary.csv', 'relearn_summary.csv']:
            try:
                os.remove(file)
            except:
                print ("couldn't remove '{}'".format(file))
        if args.mode == "train":
            os.remove("previous_queries.pkl")


if __name__ == "__main__":
    args = parser()
    logger = init_logger(args)

    if args.mode == "train":
        TrainTask(args, logger=logger)
    else:
        UnlearnTask(args, logger=logger)
