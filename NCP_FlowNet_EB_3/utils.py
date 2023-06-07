#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import tensorflow as tf
import os
import logging



def get_parameters(model):

    params = {}
    
    params['lambda'] = 1
        
    if model == 'MNIST':
        params['model'] = 'MNIST'
        params['alpha'] = .7
        params['Nmin'] = 5
        params['Nmax'] = 100
        params['h_dim'] = 256
        params['g_dim'] = 512
        params['H_dim'] = 128
        params['img_sz'] = 28
        params['channels'] = 1
        
    elif model == 'FASHIONMNIST':
        params['model'] = 'FASHIONMNIST'
        params['alpha'] = .7
        params['Nmin'] = 5
        params['Nmax'] = 100
        params['h_dim'] = 256
        params['g_dim'] = 512
        params['H_dim'] = 128
        params['img_sz'] = 28
        params['channels'] = 1
    
    elif model == 'CIFAR':
        params['model'] = 'CIFAR'
        params['alpha'] = .7
        params['Nmin'] = 5
        params['Nmax'] = 100
        params['h_dim'] = 256
        params['g_dim'] = 512
        params['H_dim'] = 128
        params['img_sz'] = 28
        params['channels'] = 3
                
    elif model == 'Gauss2D':         
        params = {}
        params['model'] = 'Gauss2D'
        params['alpha'] = .7
        params['sigma'] = 1        # std for the Gaussian noise around the cluster mean 
        params['lambda'] = 10      # std for the Gaussian prior that generates de centers of the clusters
        params['Nmin'] = 5
        params['Nmax'] = 100
        params['x_dim'] = 2
        params['h_dim'] = 256
        params['g_dim'] = 512
        params['H_dim'] = 128
        params['img_sz'] = None
        params['channels'] = None
        
    else:
        raise NameError('Unknown model '+ model)
        
    return params



def relabel(cs):
    cs = cs.copy()
    d={}
    k=0
    for i in range(len(cs)):
        j = cs[i]
        if j not in d:
            d[j] = k
            k+=1
        cs[i] = d[j]        

    return cs

        

def restore_checkpoint(ckpt_dir, state, device):
  if not tf.io.gfile.exists(ckpt_dir):
    tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
    logging.warning(f"\nNo checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state
  else:
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['step'] = loaded_state['step']
    return state


def save_checkpoint(ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'step': state['step']
  }
  torch.save(saved_state, ckpt_dir)





    



