import torch
from torch.distributions import (Dirichlet, Categorical)


class MultivariateNormal(object):
    def __init__(self, dim):
        self.dim = dim

    def sample(self, B, K, labels):
        raise NotImplementedError

    def log_prob(self, X, params):
        raise NotImplementedError

    def stats(self):
        raise NotImplementedError

    def parse(self, raw):
        raise NotImplementedError

class MixtureOfMVNs(object):
    def __init__(self, mvn):
        self.mvn = mvn

    def sample(self, B, N, K, return_gt=False):
        device = 'cpu' if not torch.cuda.is_available() \
                else torch.cuda.current_device()
        pi = Dirichlet(torch.ones(K)).sample(torch.Size([B])).to(device)
        labels = Categorical(probs=pi).sample(torch.Size([N])).to(device)
        labels = labels.transpose(0,1).contiguous()

        X, params = self.mvn.sample(B, K, labels)
        if return_gt:
            return X, labels, pi, params
        else:
            return X

    def log_prob(self, X, pi, params, return_labels=False):
        # X: [B, n, D]  (D is the h_dim, which is X after passing the pre_encoder, and not neccessarily the original X dimension)
        
        X = X.view(X.size(0), X.size(1), -1)
        ll = self.mvn.log_prob(X, params)  # [B, n, K]
        ll = ll + (pi + 1e-10).log().unsqueeze(-2)  # [B, n, K]
        if return_labels:
            labels = ll.argmax(-1)  # [B, n]
            return ll.logsumexp(-1).mean(), labels
        else:
            return ll.logsumexp(-1).mean()

    def parse(self, raw):
        return self.mvn.parse(raw)
