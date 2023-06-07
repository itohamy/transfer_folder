#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import time
import os
import torch

from NCP_FlowNet_EB.old_ncp import NeuralClustering
from data_generators import get_generator
from plot_functions import plot_avgs, plot_samples_2d, plot_samples_MNIST
from utils import relabel, get_parameters

from torch.utils.tensorboard import SummaryWriter



def main(args):

    writer = SummaryWriter()    

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
        
    model = args.model
    params = get_parameters(model)
    params['device'] = torch.device("cuda:0" if args.cuda else "cpu")
    
    print(params['device'])

    
    dpmm = NeuralClustering(params).to(params['device'])
    data_generator = get_generator(params)
    

    #define containers to collect statistics
    losses= []       # NLLs    
    accs =[]         # Accuracy of the classification prediction
    perm_vars = []   # permutation variance

    
    it=0      # iteration counter
    learning_rate = 1e-4
    weight_decay = 0.01
    optimizer = torch.optim.Adam( dpmm.parameters() , lr=learning_rate, weight_decay = weight_decay)
    
    perms = 6  # Number of permutations for each mini-batch. 
               # In each permutation, the order of the datapoints is shuffled.         
               
    batch_size = args.batch_size
    max_it = args.iterations
        
    if params['model'] == 'Gauss2D':
        if not os.path.isdir('saved_models/Gauss2D'):
            os.makedirs('saved_models/Gauss2D')
        if not os.path.isdir('figures/Gauss2D'):
            os.makedirs('figures/Gauss2D')

    elif params['model'] == 'MNIST':
        if not os.path.isdir('saved_models/MNIST'):
            os.makedirs('saved_models/MNIST')
        if not os.path.isdir('figures/MNIST'):
            os.makedirs('figures/MNIST')
    
    end_name = params['model']    
    learning_rates = {1200:5e-5, 2200:1e-5}
    
    t_start = time.time()
    itt = it
    while True:
                
            print('it', it)
            it += 1
    
            # if it == max_it:
            #     break

            # if it == 52:
            #     break
            
            dpmm.train()
                        
            # if it % args.plot_interval == 0:
                
            #     torch.cuda.empty_cache()                
            #     plot_avgs(losses, accs, perm_vars, 50, save_name='./figures/train_avgs_' + end_name + '.pdf')            
    
            #     if params['model'] == 'Gauss2D':
            #         fig_name = './figures/Gauss2D/samples_2D_' + str(it) + '.pdf'
            #         print('\nCreating plot at ' + fig_name + '\n')
            #         plot_samples_2d(dpmm, data_generator, N=100, seed=it, save_name=fig_name)    
                    
                    
            #     elif params['model'] == 'MNIST':
            #         fig_name = './figures/MNIST/samples_MNIST_' + str(it) + '.pdf'
            #         print('\nCreating plot at ' + fig_name + '\n')
            #         plot_samples_MNIST(dpmm, data_generator, N=20, seed=it, save_name= fig_name)
    
                
            # if it % 100 == 0:
            #     if 'fname' in vars():
            #         os.remove(fname)
            #     dpmm.params['it'] = it
            #     fname = 'saved_models/'+ end_name + '/'+ end_name +'_' + str(it) + '.pt'            
            #     torch.save(dpmm,fname)
    
                
            if it in learning_rates:            
                optimizer = torch.optim.Adam( dpmm.parameters() , lr=learning_rates[it], weight_decay = weight_decay)
                

            data, cs, clusters, K = data_generator.generate(None, batch_size)    
            N=data.shape[1]            
            cs = cs[None,:]
            cs = torch.tensor(cs)            
            dpmm.encode(data)  # prepare the encoded data of the batch
            
            data_Es = 0      
            MC = 0       # Marginal consistency objective
            log_pn = 0
            ll2 = 0
            for n in range(1,N):
                
                E, E_mask = dpmm(cs,n)   
                data_Es += E[:,cs[0,n]].mean()

                ll2 += (E**2.).mean()
                
                if n ==1:
                    log_pn = -E[:,cs[0,n]]
                if n >=2:                    
                    MC_batch = (log_pn - torch.log( (torch.exp(-E)*E_mask).sum(1) ) )**2                    
                    MC += MC_batch.mean() 
                    log_pn = -E[:,cs[0,n]]
            
            
            data_E = data_Es/(N-1)   # take the mean over the N-1 summed terms 
            MC *= 1/(N-2)            # take the mean over the N-2 summed terms 

            
            fake_E = dpmm.sample()
            
            l2_penalty = .1*((fake_E**2.).mean() + ll2) 
            
            fake_Em = fake_E.mean()  # mean over the mini batch234
            
            loss = data_E -fake_Em  + MC + l2_penalty  # Contrastive divergence plus marginal consistency objective 
            
            #loss = -obj + grad_reg + args.l2 * ((logp_real ** 2.).mean() + (logp_fake ** 2.).mean())
            
            writer.add_scalar("MC/train", MC, it)
            writer.add_scalar("Data E", data_E, it)
            writer.add_scalar("Fake E", fake_Em, it)
            writer.add_scalar("Diff E", data_E - fake_Em, it)


            optimizer.zero_grad()                          
            loss.backward()   
            optimizer.step()
            
            
            writer.flush()

            
            
            




            


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Neural Clustering Process')

    parser.add_argument('--model', type=str, default='Gauss2D', metavar='S',
                    choices = ['Gauss2D','MNIST'],
                    help='Generative Model: Gauss2D or MNIST (default: Gauss2D)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='batch size for training (default: 64)')
    parser.add_argument('--iterations', type=int, default=3500, metavar='N',
                    help='number of iterations to train (default: 3500)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
    parser.add_argument('--plot-interval', type=int, default=30, metavar='N',
                    help='how many iterations between training plots')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    main(args)

