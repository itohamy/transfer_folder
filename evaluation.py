import torch
import numpy as np
from plot_functions import plot_samples
from utils import compute_NMI, compute_ARI, relabel
import wandb
from collections import OrderedDict


def eval_stats(wnb, data_generator, mvn, mog, batch_size, params, net, it, stats, M=50):
    # M: number of test samples to compute stats on
    
    torch.cuda.empty_cache()
    dataname = params['dataset_name']
    net.eval()
        
    with torch.no_grad():
        
        NMI_test = 0
        ARI_test = 0
        LL_test = 0
        for i in range(M):
            data, cs_gt, _, K = data_generator.generate(N=None, batch_size=batch_size, train=False)  # data: [1, N, 2] or [1, N_sampling, 28, 28] or [1, N, 3, 28, 28]
            cs_gt = cs_gt[0, :] 
            N = data.size(1)
            cs_test = np.zeros((batch_size, N))
            
            # Get sampled clustering assignments given the data groups:
            net_output, pre_enc_output = net(data)  # [K_fixed, x_dim*2]
            pi, thetas = mvn.parse(net_output)
            ll, pred_labels = mog.log_prob(pre_enc_output, pi, thetas, return_labels=True)  # pred_labels: [B, N]; ll: [] (scalar)
            
            # Relabel pred_labels so that they appear in order (in order to compare them to the ground-truth labels)
            pred_labels_nmp = pred_labels.detach().cpu().numpy()
            for b in range(batch_size):
                cs_test[b, :] = relabel(pred_labels_nmp[b, :])

            NMI_test += compute_NMI(cs_gt, cs_test, None)
            ARI_test += compute_ARI(cs_gt, cs_test, None)
            LL_test += ll
        
        NMI_test = NMI_test / M
        ARI_test = ARI_test / M
        LL_test = LL_test / M
        
        print('\n(eval) iteration: {0}, N: {1}, NMI_test: {2:.3f}, ARI_test: {3:.3f}, LL_test: {4:.3f}'.format(it, N, NMI_test, ARI_test, LL_test))

        curr_stats = OrderedDict(it=it)
        curr_stats.update({'NMI_test': NMI_test})
        curr_stats.update({'ARI_test': ARI_test})
        curr_stats.update({'LL_test': LL_test})
        wandb.log(curr_stats, step=it)
        
        if NMI_test > stats['NMI_max']:
            stats.update({'NMI_max': NMI_test})
            stats.update({'NMI_max_it': it})
        if ARI_test > stats['ARI_max']:
            stats.update({'ARI_max': ARI_test})
            stats.update({'ARI_max_it': it})
        if LL_test > stats['LL_max']:
            stats.update({'LL_max': LL_test})
            stats.update({'LL_max_it': it})
                                    
    net.train()
    return stats
        


def plot_samples_and_histogram(wnb, data_orig, mvn, mog, cs_gt, params, net, it, N=20):
    
    if params['dataset_name'] != 'Features':
        torch.cuda.empty_cache()  
        dataname = params['dataset_name']
        net.eval()

        with torch.no_grad():
                    
            # Get sampled clustering assignments given the data groups:
            net_output, pre_enc_output = net(data_orig)  # [K_fixed, x_dim*2]
            pi, thetas = mvn.parse(net_output)
            ll, pred_labels = mog.log_prob(pre_enc_output, pi, thetas, return_labels=True)  # pred_labels: [1, N]; ll: [] (scalar)
            
            # Relabel pred_labels so that they appear in order (in order to compare them to the ground-truth labels)
            cs_test = relabel(pred_labels[0, :].detach().cpu().numpy())  # !! should be in shape [1, :]
            cs_test = np.expand_dims(cs_test, 0)
            NMI_test_sampling = compute_NMI(cs_gt, cs_test, None)
            probs = np.ones((1,))
            
            fig1, plt1 = plot_samples(params, dataname, data_orig, cs_test, probs, nmi=NMI_test_sampling, seed=it)
            image = wandb.Image(fig1)
            wnb.log({f"Plots/sampling_{it}": image}, step=it)
            plt1.clf()
                
            # Plot sampling of another permutation of the same data:
            arr = np.arange(N)
            np.random.shuffle(arr)   # permute the order in which the points are queried
            data_orig_perm2 = data_orig[:, arr, :]
            cs_gt_perm2 = cs_gt[arr]
            
            net_output2, pre_enc_output2 = net(data_orig_perm2)  # [K_fixed, x_dim*2]
            pi2, thetas2 = mvn.parse(net_output2)
            ll2, pred_labels2 = mog.log_prob(pre_enc_output2, pi2, thetas2, return_labels=True)  # pred_labels: [1, N]; ll: [1,]?
            cs_test2 = relabel(pred_labels2[0, :].detach().cpu().numpy())
            cs_test2 = np.expand_dims(cs_test2, 0)
            
            NMI_test_sampling2 = -1
            fig2, plt2 = plot_samples(params, dataname, data_orig_perm2, cs_test2, probs, nmi=NMI_test_sampling2, seed=it)
            image = wandb.Image(fig2)
            wnb.log({f"Plots/sampling_with_permuted_data_{it}": image}, step=it)
            plt2.clf()

                
        net.train()


