#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
        '''
            Input: [B * N, 28, 28]
            Output: [B * N, h]
        '''

        x = x.unsqueeze(1)   # add channel index
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class CIFAR_encoder(nn.Module):
    
    def __init__(self, params):
        super(CIFAR_encoder, self).__init__()
        
        self.conv1 = nn.Conv2d(params['channels'], 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 256)
        self.fc2 = nn.Linear(256, params['h_dim'])

    def forward(self, x):
        '''
            Input: [B * N, 3, 28, 28]
            Output: [B * N, h]
        '''

        # x = x.unsqueeze(1)   # add channel index
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


class NeuralClustering(nn.Module):
    
    def __init__(self, params):
        super(NeuralClustering, self).__init__()
        
        self.params = params
        self.previous_K = 1
        
        self.g_dim = params['g_dim']
        self.h_dim = params['h_dim']  # Used as the dim of u in the paper.
        H = params['H_dim']        
        
        self.device = params['device']

        if self.params['model'] == 'Gauss2D':
            self.h = Mixture_Gaussian_encoder(params)         
            self.q = Mixture_Gaussian_encoder(params)         
        elif self.params['model'] == 'MNIST':
            self.h = MNIST_encoder(params)  # Input: [B * N, 28, 28], Output: [B * N, h]     
            self.q = MNIST_encoder(params)  # Input: [B * N, 28, 28], Output: [B * N, h] 
        elif self.params['model'] == 'FASHIONMNIST':
            self.h = MNIST_encoder(params)  # Input: [B * N, 28, 28], Output: [B * N, h]     
            self.q = MNIST_encoder(params)  # Input: [B * N, 28, 28], Output: [B * N, h]  
        elif self.params['model'] == 'CIFAR':
            self.h = CIFAR_encoder(params)  # Input: [B * N, 3, 28, 28], Output: [B * N, h]     
            self.q = CIFAR_encoder(params)  # Input: [B * N, 3, 28, 28], Output: [B * N, h]         
        else:
            raise NameError('Unknown model '+ self.params['model'])
    
        # Input: [B, h]
        # Output: [B, g]
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
        
        # Input: [B, h+g]
        # Output: [B, 1]
        self.f = torch.nn.Sequential(
                torch.nn.Linear(self.g_dim + self.h_dim, H),
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
    
    def forward(self, data, cs, n, epsilon=0.5):
        ''' 
            Input:
            data: [B, N, 28, 28]. A batch of B points, each point is a dataset of N points with the same mixture.
            cs: [N]. Ground-truth labels of data. Labels appear in increasing order.
            n: scalar. The index of the point to be assigned.
            epsilon: number to use in the (GFN) loss calculation of each term in the sum.
            
            Output:
            logprobs_n: [B, K + 1]. Holds the log (unnormalized) probs p(c_{0:n} | x) and they differ only in the c_n assignment.
            self.logprobs_gt: [B,]. Holds the log (unnormalized) prob p(c_{0:n} | x) where all points are assigned to the correct cluster. 
            logprobs_kl: [B, K + 1]. Holds the log probability p(c_n | c_{0:n-1}, x) and they differ only in the c_n assignment.
            
            More:            
            self.Hs: [B, K, h]. Holds the sum of all points (after converting each point from x_dim to h_dim) within each cluster.
            self.Q: [B, h]. Holds the sum of all unassigned datapoints (after converting each point from x_dim to h_dim).
            Gk: [B, g]. Holds the sum of all K values of g(H_k), where the n-th point was summed to cluster k. 
        '''
                
        K = len(set(cs[:n]))  # num of already created clusters (points {0...n-1})
        logprobs_n = torch.zeros([data.shape[0], K + 1]).to(self.device) # [B, K + 1]
            
        # -------- Steps at the begining of the loss computation of one batch:
        if n == 1:
            self.batch_size = data.shape[0]
            self.N = data.shape[1]
            assert (cs == relabel(cs)).all() 
            data = self.prepare_data_x_dim(data) # [B*N, 28, 28]
            self.hs = self.h(data).view([self.batch_size, self.N, self.h_dim])   # [B, N, h] 
            self.Hs = torch.zeros([self.batch_size, 1, self.h_dim]).to(self.device) # [B, 1, h]
            self.qs = self.q(data).view([self.batch_size, self.N, self.h_dim])   # [B, N, h] 
            self.logprobs_gt = torch.zeros([self.batch_size, self.N]).to(self.device) # [B, N]. Entry n Keeps the values log(p(c_{0:n} | x))
            
            # Prepare H and U for c_0:
            self.Hs[:, 0, :] = self.hs[:, 0, :]  # [B, 1, h]. This is the H_k of the single cluster we revealed so far (point n==0 was assigned to it).
            self.Q = self.qs[:, 1:, ].sum(dim=1)    # [B, h]  Initialize U: (it will contain all points except the first point)

            # -------- Compute f(G, U) = log(p(c_{0:n} | x)) for c_0: ----------------
            Hs2 = self.Hs.clone()  # [B, K, h]. K is the number of clusters we revealed so far.
            G = self.prepare_G(Hs2) # [B, g] 
            uu = torch.cat((G, self.Q), dim=1)   # [B, g+h]. The argument for the call to f()
            self.logprobs_gt[:, 0] = torch.squeeze(self.f(uu))  # [B,]. self.f return the scalar which is the logprob of c_{0..n-1} being assigned to the right clusters.  

        # Prepare the in-edge based on the p(c_{0:n-1} | x):
        in_edge = torch.log(epsilon + torch.exp(self.logprobs_gt[:, n - 1]))  # [B,]
        
        assert self.Hs.shape[1] == K
        
        #<<<<<<<<<<<<<<<<<<<< Compute out-edges: log Sum p(c_{0:n} | x) >>>>>>>>>>>>>>>>>>>>>>>>>

        if n < self.N:  # Compute out-edge only for n=1,...,N-1
        
            # -------- Prepare U for point n: -------- 
            if n == self.N - 1:
                self.Q = torch.zeros([self.batch_size, self.h_dim]).to(self.device)  # [B, h]
            else:
                self.Q -= self.qs[:, n, ]
            
            # Compute G_k for existing clusters:
            for k in range(K):
                # Compute f(G, U) for point n (existing clusters):
                Hs2 = self.Hs.clone()  # [B, K, h]. K is the number of clusters we revealed so far.
                Hs2[:, k, :] += self.hs[:, n, :]  # Add h_i to the relevangt H_k
                Gk = self.prepare_G(Hs2) # [B, g]
                uu = torch.cat((Gk, self.Q), dim=1)  # prepare argument for the call to f()
                logprobs_n[:, k] = torch.squeeze(self.f(uu))  # [B, K + 1]. self.f return the scalar which is the logprob of n being assigned to cluster k.
                
            # Compute f(G, U) for point n (new cluster):
            Hs2 = torch.cat((self.Hs, self.hs[:, n, :].unsqueeze(1)), dim=1)  # [B, K+1, h]   
            Gk = self.prepare_G(Hs2) # [B, g]
            uu = torch.cat((Gk, self.Q), dim=1)   # [B, g+h]. The argument for the call to f()
            logprobs_n[:, K] = torch.squeeze(self.f(uu))  # [B, K + 1]. self.f return the scalar which is the logprob of n being assigned to cluster K.  

            # Update self.Hs for the next iteration:
            if cs[n] < K: # If the label of n is an existing cluster    
                self.Hs[:, cs[n], :] += self.hs[:, n, :] # [B, K, h]. K is the number of clusters we revealed so far.
            else:          # If the label of n is a new cluster   
                self.Hs = torch.cat((self.Hs, self.hs[:, n, :].unsqueeze(1)), dim=1)
                    
            # Update self.logprobs_gt with the correct value for the next n:
            self.logprobs_gt[:, n] = logprobs_n[:, cs[n]]

            # Prepare the out-edge based on the p(c_{0:n} | x):
            out_edges_sum = torch.log(epsilon + torch.exp(logprobs_n).sum(dim=1, keepdim=True))  # [B,]

            # For KL loss: get the logprobs which is log p(c_n|c_1..c_n-1, x)
            m, _ = torch.max(logprobs_n, 1, keepdim=True)  # [B, 1]
            logprobs_kl = logprobs_n - m - torch.log( torch.exp(logprobs_n - m).sum(dim=1, keepdim=True))
        
        else:  # When n == N, the only "out-edge" is equal to logR
            R = torch.tensor(10000).repeat(self.batch_size).to(self.device)   # [B,]. Fixed value used in the second term of the MC objective.
            out_edges_sum = torch.log(R)  # [B,]
            logprobs_kl = torch.zeros([self.batch_size, K + 1]).to(self.device) # [B, K + 1]
            
        return in_edge, out_edges_sum, logprobs_kl  # in_edge, out_edges_sum: [B,]; logprobs_kl: [B, K + 1]


    def prepare_data_x_dim(self, data):
        if self.params['model'] == 'Gauss2D':                
            # The data comes as a numpy vector
            data = torch.tensor(data).float().to(self.device)                    
            data = data.contiguous().view([self.batch_size * self.N, self.params['x_dim']])

        elif self.params['model'] == 'MNIST' or self.params['model'] == 'FASHIONMNIST':
            # The data comes as a torch tensor, we just move it to the device 
            data = data.to(self.device)    
            data = data.view([self.batch_size * self.N, 28, 28])
        
        elif self.params['model'] == 'CIFAR':
            # The data comes as a torch tensor, we just move it to the device 
            data = data.to(self.device)    
            data = data.view([self.batch_size * self.N, 3, 28, 28])
            
        return data # [B*N, 28, 28]
    
    
    def prepare_H(self):
        pass
    
    
    def prepare_U(self):
        pass
    
    
    def prepare_G(self, H):
        K = H.shape[1]
        H = H.view([self.batch_size * K, self.h_dim])  # [B * K, h]              
        gs = self.g(H).view([self.batch_size, K, self.g_dim]) # [B, K, g]
        G = gs.sum(dim=1)   # [B, g]
        return G