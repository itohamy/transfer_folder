#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["NCCL_P2P_DISABLE"] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["WANDB_SILENT"] = "true"

import numpy as np
import argparse
import time
# import tensorflow as tf
import torch
from data_generator import get_generator
from utils import *
from params import get_parameters
from evaluation import eval_stats, plot_samples_and_histogram
import shutil
from collections import OrderedDict
import random
from models import SetTransformer, DeepSet
from mixture_of_mvns import MixtureOfMVNs
from mvn_diag import MultivariateNormalDiag


try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False
    

def main(args):
    datasetname = args.dataset
    load_model = args.load_model
    params = get_parameters(datasetname)
    params['device'] = torch.device("cuda:0" if args.cuda else "cpu") 
    params['dataset_name'] = datasetname
    
    seed = args.seed
    set_seed(seed)

    wnb = init_wandb(args, params)
    
    batch_size = params['batch_size']
    max_it = params['max_it']
    epochs = 1
    lr = params['lr']
    device = params['device']
    plot_freq = params['plot_freq']
    net_type = params['net_type']
    h_dim = params['h_dim']
    K_fixed = params['K_fixed']
    output_dim = h_dim + 2  # This is the dim of the mixtures parameters (includes: pi, {mu_1,...mu_D}, sigma).

    # Define the model:
    if net_type == 'set_transformer':
        net_ = SetTransformer(params, h_dim, K_fixed, output_dim) #.cuda()
    elif net_type == 'deepset':
        net_ = DeepSet(params, h_dim, K_fixed, output_dim) #.cuda()
    else:
        raise ValueError('Invalid net {}'.format(net_type))
    
    net = torch.nn.DataParallel(net_, device_ids=list(range(0, torch.cuda.device_count()))).to(torch.device('cuda'))
    
    # Define the data generator:
    data_generator = get_generator(params)
    dataset_test_size = data_generator.dataset_test_size
    
    # Define learning rate and optimizers:
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    mvn = MultivariateNormalDiag(h_dim)
    mog = MixtureOfMVNs(mvn)

    it = 0
    
    # Object that stores the model info for saving:
    state = dict(optimizer=optimizer, model=net, step=0)
    
    # Create checkpoints directory
    checkpoint_dir = os.path.join('saved_models/', datasetname, 'checkpoints')
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = os.path.join('saved_models/', datasetname, 'checkpoints-meta', 'checkpoint.pth')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)

    # Load the trained model and set start_it, if required:
    if args.load_model:
        state = restore_checkpoint(checkpoint_meta_dir, state, params['device'])
        start_it = state['step']
        print('\nRestore model from iteration:', state['step'])
    else:
        start_it = it
        
    # This line helps wnb to get updated with the iteration number when loading from checkpoint:       
    wnb.log({'it': start_it}, step=start_it)
    
    # Initialize dictionary for eval stats:
    stats = {'NMI_max': 0, 'ARI_max': 0, 'ACC_max': 0, 'LL_max': -float('Inf'), 'MC_min': float('Inf'), 
             'NMI_max_it': 0, 'ARI_max_it': 0, 'ACC_max_it': 0, 'LL_max_it': 0, 'MC_min_it': 0}
    
    # ----------------------------------------------
    #              Main training loop:
    # ----------------------------------------------
    
    print('start_it:', start_it)
    print('max_it:', max_it)
    print('Start training.') 
    
    for it in range(start_it, max_it):
        
        if it == int(0.5*max_it):
            optimizer.param_groups[0]['lr'] *= 0.1
                        
        # Evaluate the model periodically:
        if plot_freq != -1 and it % plot_freq == 0:
            # print('\nPloting samples, compute NMI, ARI, LL, iteration ' + str(it) + '.. \n')   
            
            # NMI, ARI, LL.
            # data, cs_gt, clusters, K = data_generator.generate(N=None, batch_size=batch_size, train=False)  # data: [1, N, 2] or [1, N_sampling, 28, 28] or [1, N, 3, 28, 28]
            stats = eval_stats(wnb, data_generator, mvn, mog, batch_size, params, net, it, stats, M=dataset_test_size//batch_size)
            
            # Plots. Here we must use N=20 because we need to plot the results:
            data, cs_gt, clusters, K = data_generator.generate(N=20, batch_size=1, train=False)  # data: [1, N, 2] or [1, N, 28, 28] or [1, N, 3, 28, 28]            
            plot_samples_and_histogram(wnb, data, mvn, mog, cs_gt[0, :], params, net, it, N=20)
            
        # Save the model periodically:
        if it % 1000 == 0 and it > 1:
            print('\Saving model.. \n') 
            save_model(state, it, net, optimizer, checkpoint_dir, checkpoint_meta_dir)
  
        # Generate one batch for training
        data, cs, clusters, K = data_generator.generate(N=None, batch_size=batch_size, train=True, unsup=False)
        
        N = data.shape[1]
        
        # Prepare label-predictions tensor
        cs_pred_train = np.zeros((batch_size, N))
        
        # Training of one point: FW and Backprop of one batch.
        # (Each training step includes a few permutations of the data order)   
        net.train()
        
        # Forward step
        net_output, pre_enc_output = net(data)  # [K_fixed, h_dim*2]
        pi, thetas = mvn.parse(net_output)
        ll, pred_labels = mog.log_prob(pre_enc_output, pi, thetas, return_labels=True)  # pred_labels: [B, N]; ll: [] (scalar)
        
        # Relabel pred_labels so that they appear in order (in order to compare them to the ground-truth labels)
        pred_labels_nmp = pred_labels.detach().cpu().numpy()
        for b in range(batch_size):
            cs_pred_train[b, :] = relabel(pred_labels_nmp[b, :])
        
        loss = -ll
        loss = loss.mean()  # Average on outputs from all devices
        
        loss.backward()    # this accumulates the gradients for each permutation
        optimizer.step()      # the gradients used in this step are the sum of the gradients for each permutation 
        optimizer.zero_grad()    
        
        NMI_train = compute_NMI(cs[0, :], cs_pred_train, None)
        ARI_train = compute_ARI(cs[0, :], cs_pred_train, None)           
                
        # Store statistics in wandb:
        sts = update_stats_train(it, N, loss, NMI_train, ARI_train)  # stats.update({'train_acc1': acc_train})
        wandb.log(sts, step=it)

        if it % 10 == 0:
            print('\n(train) iteration: {0}, N: {1}, NMI_train: {2:.3f}, ARI_train: {3:.3f}, Loss: {4:.3f}'.format(it, N, NMI_train, ARI_train, loss))

        it += 1
        
    # Print avg metrics:
    print('\n\n * Best stats: \n',
            'Max NMI (test): {0:.3f} (on iter:) {1:.3f}'.format(stats['NMI_max'], stats['NMI_max_it']), '\n',
            'Max ARI (test): {0:.3f} (on iter:) {1:.3f}'.format(stats['ARI_max'], stats['ARI_max_it']), '\n')
    
    
def set_seed(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    
def init_wandb(args, params):
    if has_wandb:
        wnb = wandb.init(entity='bgu_cs_vil', project="NCP_EB", name=args.experiment, config=args)
        wnb.log_code(".")  # log source code of this run
        wnb.config.update(params)
    else:
        wnb = None
        print("Problem with initiating wandb.")
    
    return wnb
   
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Neural Clustering Process')

    parser.add_argument('--dataset', type=str, default='Gauss2D', metavar='S',
                    choices = ['Gauss2D','MNIST', 'FASHIONMNIST', 'CIFAR', 'Features', 'tinyimagenet'],
                    help='Generative Model: Gauss2D or MNIST (default: Gauss2D)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=123, metavar='S',
                    help='random seed (default: 123)')
    parser.add_argument('--show-histogram', action='store_true', default=False,
                    help='flag for analyzing a trained model')
    parser.add_argument('--load-model', action='store_true', default=False,
                    help='flag for loading model or start from scratch')       
    parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                   help='name of wandb experiment')   
        
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    if not args.load_model:
        # Remove saved models
        model_dir = 'saved_models'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        shutil.rmtree(model_dir)
    
    main(args)

