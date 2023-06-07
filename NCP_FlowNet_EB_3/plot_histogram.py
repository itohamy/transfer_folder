import numpy as np
from ncp_prob_computer import NCP_prob_computer
from ncp_sampler import NCP_Sampler
import torch
import matplotlib.pyplot as plt
from utils import relabel


        
def histogram_for_data_perms(data, dpmm, N = 20, perms = 100):
    '''
    data: one small dataset with N images. Shape: [1, N, channels, img_sz, img_sz]
    dpmm: a trained model
    N: number of images to cluster
    perms: number of data-order permutations to check. (We compute the probability of the most likely assignment given different data orders).
    '''
    
    probs_for_histogram = np.zeros(perms) # Stores the probabilities of getting the most likely assignmnet, given each data-order permutation
    
    # Find the most likely clustering of "data":
    ncp_sampler = NCP_Sampler(dpmm, data)
    S = 5000   # number of samples 
    css, probs = ncp_sampler.sample(S) # css: [M, N] probs: [M], where M is the number of succeeded samples.
    most_likely_clstr = css[np.argmax(probs), :]  # [N,]
    prob_most_likely_clstr = np.max(probs)  # scalar
        
    # Sanity check: compute the probability of getting "most_likely_clstr" using the original data order (no permutation):
    prob_computer = NCP_prob_computer(dpmm, data)
    prob_orig_order = prob_computer.compute(most_likely_clstr)
    print('\nProbability of most-likely assignment, computation vs. sampler:', prob_orig_order, prob_most_likely_clstr, '\n')
    
    # Sample "perms" permutations of data and store their probability result from the model:
    for p in range(perms):
        # Draw a permutation:
        arr = np.arange(N)
        np.random.shuffle(arr)   
        data_perm = data[:, arr, :] # permute the order of the data and the most likely assignment
        most_likely_clstr_perm = most_likely_clstr[arr]
        
        # Compute the probability of getting "most_likely_clstr_perm" with this data-order permutation:
        prob_computer = NCP_prob_computer(dpmm, data_perm)
        probs_for_histogram[p] = prob_computer.compute(most_likely_clstr_perm)
        
    # Create histogram of all probabilities: 
    fig = plt.figure(3, figsize=(15, 8))
    plt.clf()
    plt.hist(probs_for_histogram)
    
    return fig, plt, most_likely_clstr, prob_most_likely_clstr


def plot_best_clustering(data, css, prob_css):
    '''
    data: one small dataset with N images in the original order. Shape: [1, N, 28, 28] 
    css: the most likely clustering for data. [N,]
    prob_css: probability of the most likely clustering. Scalar.
    '''
    # Plot the most-likely clustering and the original data order
    
    rows = 1 # number of samples to show (e.g. we show the first 5 samples from css)
    W = 10
    N = data.shape[1]
    css = np.expand_dims(css, axis=0)

    fig = plt.figure(3, figsize=(15,8))
    plt.clf()
    
    step = 0.074
    
    for i in range(N):
        plt.subplot(W+1,26,i+1+1)
        plt.imshow(data[0,i,:,:], cmap='gray')
        plt.xticks([])
        plt.yticks([])
    
    fontsize=25    
    plt.gcf().text(0.35, .9, str(N)+ ' Observations', fontsize=fontsize)
    plt.gcf().text(0.35, 0.76, 'Most-likely Cluster Sample', fontsize=fontsize)
        
    for w in range(0, rows):
        it = css[w,:]
        K = len(set(it))
        
        dat = {}
        for k in range(K):
            dat[k]=data[0,np.where(it==k)[0],:,:]

        fontsize=15
        strtext = 'K = ' + str(K) + '  Pr: ' + '{0:.2f}'.format(prob_css) 
        plt.gcf().text(0.03, 0.63-(w-1)*step, strtext, fontsize=fontsize)
        
        i= (w+2)*26
        for k in range(K):
            for j in range(len(dat[k])):
                plt.subplot(W+1,26,i+1+1)                    
                plt.imshow(dat[k][j,:,:], cmap='gray')
                plt.xticks([])
                plt.yticks([])
                i+=1
            i+=1
    
    return fig, plt
