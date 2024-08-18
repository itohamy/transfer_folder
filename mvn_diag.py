from mixture_of_mvns import MultivariateNormal
import torch
import torch.nn.functional as F
import math

class MultivariateNormalDiag(MultivariateNormal):
    def __init__(self, dim):
        super(MultivariateNormalDiag, self).__init__(dim)

    def sample(self, B, K, labels):
        N = labels.shape[-1]
        device = labels.device
        mu = -4 + 8*torch.rand(B, K, self.dim).to(device)
        sigma = 0.3*torch.ones(B, K, self.dim).to(device)
        eps = torch.randn(B, N, self.dim).to(device)

        rlabels = labels.unsqueeze(-1).repeat(1, 1, self.dim)
        X = torch.gather(mu, 1, rlabels) + \
                eps * torch.gather(sigma, 1, rlabels)
        return X, (mu, sigma)

    # Only here we compute the ll using thetas (mu, sigma) and pi
    def log_prob(self, X, params):
        # X: [B, n, D] (D is the h_dim, which is X after passing the pre_encoder, and not neccessarily the original X dimension)
        mu, sigma = params
        dim = self.dim
        X = X.unsqueeze(2)  # [B, n, 1, D]
        mu = mu.unsqueeze(1)  # [B, 1, K, D]
        sigma = sigma.unsqueeze(1)  # [B, 1, K, 1]
        diff = X - mu  # [B, n, K, D]
        ll = -0.5*math.log(2*math.pi) - sigma.log() - 0.5*(diff.pow(2)/sigma.pow(2))  # [B, n, K, 2]
        ll_sum = ll.sum(-1)  # [B, n, K]
        return ll_sum

    def stats(self, params):
        mu, sigma = params
        I = torch.eye(self.dim)[(None,)*(len(sigma.shape)-1)].to(sigma.device)
        cov = sigma.pow(2).unsqueeze(-1) * I
        return mu, cov

    def parse(self, raw):
        # raw: [B, K, dim_output==D+2]
        pi = torch.softmax(raw[...,0], -1)  # [B, K]
        mu = raw[...,1:1+self.dim]  # [B, K, D]
        sigma = F.softplus(raw[...,1+self.dim:]) # [B, K]
        return pi, (mu, sigma)
