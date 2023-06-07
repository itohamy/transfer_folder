#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F





from utils import relabel

class MNIST_encoder(nn.Module):
    
    def __init__(self, params):
        
        super(MNIST_encoder, self).__init__()
        
        
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 256)
        self.fc2 = nn.Linear(256, params['h_dim'])

    def forward(self, x):
        
        x = x.unsqueeze(1)   # add channel index
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x




class Mixture_Gaussian_encoder(nn.Module):
    
    def __init__(self, params):
        
        super(Mixture_Gaussian_encoder, self).__init__()
        
        H = params['H_dim']
        self.h_dim = params['h_dim']        
        self.x_dim = params['x_dim']
        
        self.h = torch.nn.Sequential(
                torch.nn.Linear(self.x_dim, H),
                torch.nn.PReLU(),
                torch.nn.Linear(H, H),
                torch.nn.PReLU(),    
                torch.nn.Linear(H, H),
                torch.nn.PReLU(),
                torch.nn.Linear(H, H),
                torch.nn.PReLU(),
                torch.nn.Linear(H, self.h_dim),
                )

    def forward(self, x):
        
        return self.h(x)




class AggregateClusteredSum(nn.Module):
    def __init__(self, params):    
        super(AggregateClusteredSum, self).__init__()
        
        self.h_dim = params['h_dim']
        self.g_dim = params['g_dim']
        H = params['H_dim']     
        self.device = params['device']
        
        self.g = torch.nn.Sequential(
                torch.nn.Linear(self.h_dim, H),
                torch.nn.PReLU(),    
                torch.nn.Linear(H, H),
                torch.nn.PReLU(),
                torch.nn.Linear(H, H),
                torch.nn.PReLU(),    
                torch.nn.Linear(H, H),
                torch.nn.PReLU(),
                torch.nn.Linear(H, H),
                torch.nn.PReLU(),
                torch.nn.Linear(H, self.g_dim),
                )

    def forward(self, hs,cs,n):
        ''' Input:
            hs:  [batch_size, N, dim_h]
            cs:  [1,N], [batch_size, N]  torch tensor with non-negative integer entries
        '''
        
        cs = cs.clone()
        assert len(cs.shape) == 2                       
        cs[:,n:] = -1
            
        K = int(cs.max()) + 1
        batch_size = hs.shape[0]


        if cs.shape[0] > 1:                    
            many_cs_patterns = True
            Ks, _ = cs.max(dim=1)
        else:
            many_cs_patterns = False

        
        G  = torch.zeros([batch_size,   K+1, self.g_dim]).to(self.device)        
        H  = torch.zeros([batch_size, 2*K+1, self.h_dim ]).to(self.device)        
        
        if many_cs_patterns:
            # gs_mask allows us to transfer the spasity pattern of H  to gs G
            gs_mask = torch.zeros([batch_size, 2*K+1, 1]).to(self.device)
        
        
        for k in range(K):            
            mask = 1.0*(cs==k).to(self.device) 
            mask = mask[:,:,None]
            
            H[:,k,:] = (hs*mask).sum(1)            
            H[:,k+K,:] = H[:,k,:]  + hs[:,n,:]

            if many_cs_patterns:
                gs_mask[:,k] = torch.any((cs==k),dim=1, keepdim=True)*1.0
                gs_mask[:,k+K] = gs_mask[:,k]

        H[:,2*K,:] =  hs[:,n,:]
        
        if many_cs_patterns:
            gs_mask[:,2*K] = 1.0 
        
        H = H.view([batch_size*(2*K+1), self.h_dim])
        gs = self.g(H).view([batch_size, 2*K+1, self.g_dim])
        if many_cs_patterns:
            gs = gs_mask*gs
        
        
        for k in range(K):            
            inds = torch.tensor([True]*K + [False]*(K+1)  )
            inds[k] = False
            inds[k+K] = True
            G[:,k,:] = gs[:,inds,:].sum(1)
        
        inds = torch.tensor([True]*K + [False]*K +[True] )
        G[:,K,:] =  gs[:,inds,:].sum(1)                        
        
        if many_cs_patterns:
            G[:,:K] = G[:,:K]*gs_mask[:,:K]
        
        G_mask = torch.ones([batch_size,K+1]).to(self.device)
        
        if many_cs_patterns:
            for k in range(K-1):            
                which_k = (Ks==k)                            
                G[which_k,k+1,:] = G[which_k,-1,:]
                G[which_k,-1,:] = 0                
                G_mask[which_k,k+2:] = 0
            
       
        return G, G_mask
        


        
class AggregateUnclusteredSum(nn.Module):           
    def __init__(self, params):    
        super(AggregateUnclusteredSum, self).__init__()

    def forward(self,hs,n):        
        Q = hs[:,n:,].sum(dim=1)     #[batch_size,h_dim]        
        return Q
    
    


class NeuralClustering(nn.Module):
    
    
    def __init__(self, params):
        
        super(NeuralClustering, self).__init__()
        
        self.params = params
        self.previous_n = 0
        self.previous_K=1
        
        self.g_dim = params['g_dim']
        self.h_dim = params['h_dim']
        H = params['H_dim']        
        
        self.device = params['device']


        if self.params['model'] == 'Gauss2D':
            self.h = Mixture_Gaussian_encoder(params)         
        elif self.params['model'] == 'MNIST':
            self.h = MNIST_encoder(params)                     
        else:
            raise NameError('Unknown model '+ self.params['model'])

        
        self.agg_clustered = AggregateClusteredSum(params)
        self.agg_unclustered = AggregateUnclusteredSum(params)
        
        self.E = torch.nn.Sequential(
                torch.nn.Linear(self.g_dim +self.h_dim, H),
                torch.nn.PReLU(),    
                torch.nn.Linear(H, H),                
                torch.nn.PReLU(),    
                torch.nn.Linear(H, H),
                torch.nn.PReLU(),
                torch.nn.Linear(H, H),
                torch.nn.PReLU(),
                torch.nn.Linear(H, H),
                torch.nn.PReLU(),
                torch.nn.Linear(H, 1, bias=False),
                )
        

        
        
    def encode(self,data):
            self.batch_size = data.shape[0]
            self.N = data.shape[1]
            
            if self.params['model'] == 'Gauss2D':
                # The data comes as a numpy vector
                data = torch.tensor(data).float().to(self.device)                    
                data = data.reshape([self.batch_size*self.N, self.params['x_dim']])

            elif self.params['model'] == 'MNIST':
                # The data comes as a torch tensor, we just move it to the device 
                data = data.to(self.device)    
                data = data.view([self.batch_size*self.N, 28,28])
                                
            
            self.hs = self.h(data).view([self.batch_size,self.N, self.h_dim])            


    
    def sample(self):
        
        c_samples = torch.zeros([self.batch_size,self.N], dtype=torch.int32)        
        prob_samples = torch.zeros([self.batch_size,self.N], dtype=torch.int32)        
        E_samples = torch.zeros(self.batch_size,self.N-1).to(self.device)
        B = self.batch_size                        
        
        
        for n in range(1,self.N):
            #print(n)
            E, G_mask = self.forward(c_samples,n)       
            m,_ = torch.min(E,1, keepdim=True)        #[batch_size,1]            
            ii =  torch.multinomial(torch.exp(-E + m)*G_mask , num_samples=1).squeeze()
            c_samples[:,n] = ii 
            E_samples[:,n-1] = E[range(B),ii]
            

        return E_samples
        # Normalize
        # m,_ = torch.max(logprobs,1, keepdim=True)        #[batch_size,1]
        # logprobs = logprobs - m - torch.log( torch.exp(logprobs-m).sum(dim=1, keepdim=True))
            
            
        
    def forward(self, cs, n):
        
        ''' Input:
            cs:  torch tensor [1,N] or [batch_size, N]  with non-negative integer entries
        '''

             
        # n =1,2,3..N-1   (the first point has n=0, c=0 and does not need to be clustered)
        # elements with index below or equal to n-1 are already assigned
        # element with index n is to be assigned. 
        # the elements from the n+1-th are not assigned

        assert n >0
        K = cs[:,:n].max().item() + 1
        #K = len(set(cs[:,:n].flatten() ))  # num of already created clusters

        
        #assert (cs==relabel(cs)).all()            
        B = self.batch_size                        
        G, G_mask = self.agg_clustered(self.hs,cs,n)         # G [batch_size,K+1, g_dim], G_mask [self.batch_size,K+1]
        Q = self.agg_unclustered(self.hs,n)
        Q = Q[:,None,:].expand([B,K+1,self.h_dim])

        uu = torch.cat((G,Q), dim=2)  # prepare argument for the call to E()
        uu = uu.view([B*(K+1), self.g_dim + self.h_dim ] )        
        
        E = self.E(uu).view([B,K+1]) 
        

        return E, G_mask



