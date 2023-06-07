#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import time
import os
import tensorflow as tf
import torch
from ncp import NeuralClustering
from data_generators import get_generator
from plot_functions import plot_avgs, plot_samples_2d, plot_samples
from plot_histogram import histogram_for_data_perms, plot_best_clustering
from utils import relabel, get_parameters
import shutil
from utils import save_checkpoint, restore_checkpoint
import globals


def main(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
        
    model = args.model
    params = get_parameters(model)
    params['device'] = torch.device("cuda:0" if args.cuda else "cpu")
     
    # Define the model:
    dpmm = NeuralClustering(params).to(params['device'])
    
    # Definr the data generator:
    data_generator = get_generator(params)
    
    # Define more params for the training:
    it = 0   
    learning_rate = 1e-4
    weight_decay = 0.01
    optimizer = torch.optim.Adam(dpmm.parameters() , lr=learning_rate, weight_decay = weight_decay)
    batch_size = args.batch_size
    max_it = args.iterations  # number of iterations
    N_sampling = args.N_sampling
    datasetname = params['model']
    analyze = args.analyze  # A flag for analyzing a trained model (histogram)
    lamda = params['lambda']
    
    # Object that stores the model info for saving:
    state = dict(optimizer=optimizer, model=dpmm, step=0)
    
    # Create checkpoints directory
    checkpoint_dir = os.path.join('saved_models/', datasetname, 'checkpoints')
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = os.path.join('saved_models/', datasetname, 'checkpoints-meta', 'checkpoint.pth')
    tf.io.gfile.makedirs(checkpoint_dir)
    tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))

    # Load the trained model if required:
    state = restore_checkpoint(checkpoint_meta_dir, state, params['device'])
    print('\nRestore model from iteration:', state['step'])
    
    # learning_rates = {600:1e-4, 1200:5e-5, 2200:1e-5}
    learning_rates = {1200:5e-5, 2200:1e-5}
    
    # Store configs in neptune:
    globals.run["Params/dataset"] = params['model']
    globals.run["Params/batch_sz"] = args.batch_size
    globals.run["Params/h_dim"] = params['h_dim']
    globals.run["Params/u_dim"] = params['h_dim']
    globals.run["Params/g_dim"] = params['g_dim']
    globals.run["Params/lambda"] = params['lambda']
    globals.run["Params/N_sampling"] = args.N_sampling
    
    # -------------------------------------
    #         Main training loop:
    # -------------------------------------
    
    while it < max_it:
                
        it += 1

        dpmm.train()
                   
        # Plot samples periodically: 
        if it % args.plot_interval == 0 or it == 1:
            print('\nPloting sample, iteration ' + str(it) + '.. \n')          
            plot_samples_periodically(params, dpmm, it, data_generator, N_sampling, analyze, params['device'])
                     
        # Save the model periodically:
        if it % 100 == 0:
            print('\Saving model.. \n') 
            save_model(state, it, dpmm, optimizer, checkpoint_dir, checkpoint_meta_dir)
            
        # Define the optimizer based on iteration number:
        if it in learning_rates:            
            optimizer = torch.optim.Adam(dpmm.parameters(), lr=learning_rates[it], weight_decay=weight_decay)

        # Generate one batch for training
        data, cs, clusters, K = data_generator.generate(None, batch_size)    
        N = data.shape[1]
        
        print('Iteration:' + str(it) + ' N:', str(N))          
        
        MC_loss = 0    # Marginal consistency (MC) objective
        # log_pn = 0  # Used as the "in-edges" in the MC loss. The vakue is equal to: (- unnormalized logprob of p(c_{0:n-1} | x)). 
        kl_loss = 0   # Holds the KL loss as computed in the original NCP paper.
        ll2 = 0
        data_E = torch.zeros(batch_size, 1).to(params['device'])
        epsilon = 1e-5
         
        # Permute the order of the batch (needed when we do more then one permutation in a training step)      
        # data, cs = permute_data(N, data, cs)
        
        # make cluster labels appear in cs[] in increasing order
        cs = relabel(cs)
        cs = cs[None, :]  # [1, N]
        cs = torch.tensor(cs) 
        
        # Prepare the first encoding of the batch:
        dpmm.encode(data)  # Data is now in shape [B, N, h_dim]  

        # check if we should add the first "in-edge" for MC loss
        # Compute f(G, U) = unnormalized logprob of p(c_0 | x) for the first term of MC sum.
        # log_pn = - dpmm.logprob_c_0()  # [B, 1]
        
        # FW step:
        for n in range(1, N):
            E, E_mask = dpmm(cs, n)   # E is [B, K + 1]. This is the unnormalized logprob of p(c_{0:n} | x).
            
            # ----- J Loss: -----
            # if n == N - 1:
            #     data_E = E[:, cs[0, n]].mean()  # scalar. E[:, cs[0, n]] is [B, 1], it's the unnormalized logprob of p(c_{0:N} | x) using the ground-truth label for the N-th point.         
             
            # ----- P Loss: -----
            if n == N - 1:            
                m, _ = torch.min(E, 1, keepdim=True)  
                E_n = torch.unsqueeze(E[:, cs[0, n]], 1)
                data_E = (- E_n + m - torch.log((torch.exp(- E + m) * E_mask).sum(dim=1, keepdim=True))) ** 2  # [B, 1]. logprob of p(c_N | c_{0:N}, x) using the ground-truth label for the N-th point.
                data_E_unlog = (torch.exp(- E + m) * E_mask)[:, cs[0, n]] / (torch.exp(- E + m) * E_mask).sum(dim=1, keepdim=True)
                
            # ----- MC Loss: -----
            if n == 1:
                # m, _ = torch.min(E, 1, keepdim=True)    # [B, 1] 
                log_pn = - torch.unsqueeze(E[:, cs[0, n]], 1)   # + m  # [B, 1], unnormalized logprob of p(c_{0:n} | x)
                
            elif n >= 2: 
                m, _ = torch.min(E, 1, keepdim=True)    # [B, 1]
                MC_n_term = (log_pn - torch.log((torch.exp(- E + m) * E_mask).sum(1))) ** 2  # { unnormalized logprob(c_{0:n-1}|x) - log(sum_{c_n}(exp(unnormalized logprob(c_{0:n}|x)))) }^2                
                MC_n_term_0 = (torch.log(epsilon + torch.exp(log_pn)) - torch.log(epsilon + (torch.exp(-E) * E_mask).sum(1))) ** 2           
                MC_loss += MC_n_term.mean() 
                log_pn = - torch.unsqueeze(E[:, cs[0, n]], 1)  # + m  # [B, 1], unnormalized logprob of p(c_{0:n} | x)
                
                # if n == N - 1:
                #     R = torch.tensor(10000).repeat(batch_size).to(params['device'])   # [B,]. Fixed value used in the second term of the MC objective.
                #     last_MC_term = ((torch.log(epsilon + torch.exp(log_pn)) - torch.log(R)) ** 2).mean() 
                #     MC_loss += last_MC_term
                
                # if it == 150 and n > N - 5:
                #     print('n: ', n)
                #     print('cs[n]: ', cs[0, n])
                #     print('E: ', E)
                #     print('in edges (logprob): ', log_pn)
                #     print('sum out edges (log-sum-exp): ', torch.log((torch.exp(- E + m) * E_mask).sum(1)))
                #     print('in edges (probs): ', torch.exp(log_pn))
                #     print('out edges (probs): ', (torch.exp(- E + m) * E_mask))
                #     if n == N - 1:
                #         print('data_E: ', data_E)
                #         print('data_E_unlog: ', data_E_unlog)
                #         print('last_MC_term: ', last_MC_term)
                #         1/0
                                
            # ----- KL loss: get the logprobs which is log p(c_n|c_1..c_n-1, x) ----------------
            E = E * E_mask
            m, _ = torch.min(E, 1, keepdim=True)    # [B, 1]                  
            logprobs_kl = ((-E + m) * E_mask) - torch.log((torch.exp(-E + m) * E_mask).sum(dim=1, keepdim=True))
            c = cs[0, n] # The ground-truth cluster of the n-th point (which is similar in all B datasets)
            kl_loss -= logprobs_kl[:, c].mean() # The loss is minus the value in the relevant cluster assignment in logprobs.
            # ---------------------------------------------------------------------------------------------------
 
        # data_E = data_Es/(N - 1)   # take the mean over the N-1 summed terms 
        
        MC_loss *= 1 / (N - 2)            # take the mean over the N-2 summed terms 
        
        # fake_E = dpmm.sample(it)[:, N - 2]
        # fake_E = dpmm.sample(it)  # [B,1]]. logprob of p(c_N | c_{0:N}, x) using the sampled label for the N-th point. 
        
        # l2_penalty = .1 * ((fake_E ** 2.).mean() + (data_E ** 2.).mean())
        # l2_penalty = .1 * ((fake_E ** 2.).mean() + (data_E ** 2.).mean())
        
        # fake_E = fake_E.mean()  # scalar. Mean over the minibatch
        data_E = data_E.mean()  # scalar. Mean over the minibatch
        
        # J_loss = data_E - fake_E
        P_loss = data_E # - fake_E
        
        # loss = kl_loss  # J_loss  + MC_loss + l2_penalty  # Contrastive divergence plus marginal consistency objective 
        loss = MC_loss - P_loss # P_loss + MC_loss # + l2_penalty  # Contrastive divergence plus marginal consistency objective 
        #loss = -obj + grad_reg + args.l2 * ((logp_real ** 2.).mean() + (logp_fake ** 2.).mean())

        optimizer.zero_grad()                          
        loss.backward()   
        optimizer.step()
        
        # Store statistics in neptune:
        globals.run["Training/KL_loss"].append(kl_loss.item() / N)
        globals.run["Training/MC_loss"].append(MC_loss.item())
        # globals.run["Training/J_loss"].append(J_loss.item())
        globals.run["Training/P_loss"].append(P_loss.item())
        # globals.run["Training/l2_penalty"].append(l2_penalty.item())
        globals.run["Training/loss"].append(loss.item())
        
        # ---------- !!!!!!!! BEGIN OLD MAIN: -----------
        
        # for n in range(1, N + 1):
        #     in_edge, out_edges_sum, logprobs_kl  = dpmm(data, cs, n) # # both are in shape [B,]
        #     # MC Objective:
        #     this_mc_loss += torch.square(in_edge - out_edges_sum)  # [B,] 
            
        #     # KL objective:
        #     if n < N:  # when n == 1,...,N-1:
        #         c = cs[n] # The ground-truth cluster of the n-th point (which is similar in all B datasets)
        #         # accuracies[n - 1, perm] = np.sum(np.argmax(logprobs_kl.detach().to('cpu').numpy(), axis=1)==c) / logprobs_kl.shape[0]            
        #         this_kl_loss -= logprobs_kl[:, c].mean() # The loss is minus the value in the relevant cluster assignment in logprobs.
            
        # this_mc_loss = this_mc_loss.mean() 
            
        # # Compute the loss of this permutation:
        # this_total_loss = this_mc_loss  #(this_kl_loss + (lamda * this_mc_loss))/ N 
        # this_total_loss.backward()    # this accumulates the gradients for each permutation
        
        # loss_values[perm] = this_total_loss
        # mc_loss += this_mc_loss
        # kl_loss += this_kl_loss
        # loss += this_total_loss
        
        # perm_vars.append(loss_values.var())
        # losses.append(loss.item() / (N * perms))
        
        # # Store statistics in neptune:
        # globals.run["Training/N_values"].append(N)
        # globals.run["Training/K_values"].append(K)
        # globals.run["Training/permutations_variance"].append(loss_values.var())
        # globals.run["Training/loss"].append(loss.item() / N)
        # globals.run["Training/mc_loss"].append(mc_loss.item() / N)
        # globals.run["Training/kl_loss"].append(kl_loss.item() / N)
        
        # optimizer.step()      # the gradients used in this step are the sum of the gradients for each permutation 
        # optimizer.zero_grad()    

        # print('{0:4d}  N:{1:2d}  K:{2}  Mean NLL:{3:.3f}  Mean Permutation Variance: {4:.7f}  Mean Time/Iteration: {5:.1f}'\
        #         .format(it, N, K , np.mean(losses[-50:]), np.mean(perm_vars[-50:]), (time.time()-t_start)/(it - itt)    ))    

        # # The memory requirements change in each iteration according to the random values of N and K. 
        # # If both N and K are big, an out of memory RuntimeError exception might be raised.
        # # When this happens, we capture the exception, reduce the batch_size to 3/4 of its value, and try again.
        
        # # except RuntimeError:
        # #     bsize = int(.75*data.shape[0])
        # #     if bsize > 2:
        # #         print('RuntimeError handled  ', 'N:', N, ' K:', K, 'Trying batch size:', bsize)
        # #         data = data[:bsize,:,:]
        # #     else:
        # #         break
    
        # ------- !!!!!!!! END OLD MAIN -----------


def plot_samples_periodically(params, dpmm, it, data_generator, N_sampling, analyze, device):
    torch.cuda.empty_cache()  
    datasetname = params['model']
    dpmm.eval()
    
    if datasetname == 'Gauss2D':
        _, __, fig, plt = plot_samples_2d(dpmm, data_generator, device, N=100, seed=it)
        globals.run['Sampling/iteration_' + str(it)].append(fig)
        plt.clf()    
        
    else:                
        # Generate one small dataset with N images:
        data, _, _, _ = data_generator.generate(N = N_sampling, batch_size=1)  # data: [1, N_sampling, 28, 28] or [1, N, 3, 28, 28]
        fig1, plt1 = plot_samples(datasetname, dpmm, data, N = N_sampling, seed=it)
        globals.run['Sampling/iteration_' + str(it)].append(fig1)
        plt1.clf()
        
        # Plot a second sampling of the same data order:
        fig1_, plt1_ = plot_samples(datasetname, dpmm, data, N = N_sampling, seed=it)
        globals.run['Sampling/iteration_' + str(it)].append(fig1_)
        plt1_.clf()

        # Plot sampling of another permutation of the same data:
        arr = np.arange(N_sampling)
        np.random.shuffle(arr)   # permute the order in which the points are queried
        data_perm2 = data[:, arr, :]
        fig2, plt2 = plot_samples(datasetname, dpmm, data_perm2, N = N_sampling, seed=it)
        globals.run['Sampling/iteration_' + str(it)].append(fig2)
        plt2.clf()
        
        # Analyze the sample results of different data orders (histogram of probabilities): 
        if analyze:
            fig3, plt3, most_likely_clstr, prob_most_likely_clstr = histogram_for_data_perms(data, dpmm, N=N_sampling, perms=500)
            globals.run['Sampling/histogram/iteration_' + str(it)].append(fig3)
            plt3.clf()
            
            fig4, plt4 = plot_best_clustering(data, most_likely_clstr, prob_most_likely_clstr)
            globals.run['Sampling/histogram/iteration_' + str(it)].append(fig4)
            plt4.clf()
        
    dpmm.train()


def save_model(state, it, dpmm, optimizer, checkpoint_dir, checkpoint_meta_dir):
    state['step'] = it
    state['model'] = dpmm
    state['optimizer'] = optimizer
    save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{it}.pth'), state)
    save_checkpoint(checkpoint_meta_dir, state)


def permute_data(N, data, cs):
    arr = np.arange(N)
    np.random.shuffle(arr)   
    cs = cs[arr]
    data = data[:, arr, :]    
    return data, cs
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Neural Clustering Process')

    parser.add_argument('--model', type=str, default='Gauss2D', metavar='S',
                    choices = ['Gauss2D','MNIST', 'FASHIONMNIST', 'CIFAR'],
                    help='Generative Model: Gauss2D or MNIST (default: Gauss2D)')
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                    help='batch size for training (default: 64)')
    parser.add_argument('--iterations', type=int, default=3500, metavar='N',
                    help='number of iterations to train (default: 3500)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
    parser.add_argument('--plot-interval', type=int, default=30, metavar='N',
                    help='how many iterations between training plots')
    parser.add_argument('--N-sampling', type=int, default=20, metavar='N',
                    help='N data points when sampling (default: 20)')
    parser.add_argument('--analyze', action='store_true', default=False,
                    help='flag for analyzing a trained model')
    parser.add_argument('--load-model', action='store_true', default=False,
                    help='flag for loading model or start from scratch')       
        
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    globals.initialize() 
    
    if not args.load_model:
        # Remove saved models
        model_dir = 'saved_models'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        shutil.rmtree(model_dir)
    
    main(args)

