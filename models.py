from modules import *
from pre_encoders import get_encoder



class DeepSet(nn.Module):
    def __init__(self, params, dim_input, num_outputs, dim_output, dim_hidden=128):
        super(DeepSet, self).__init__()
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        
        self.pre_enc = get_encoder(params)
        
        self.enc = nn.Sequential(
                nn.Linear(dim_input, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden))
        
        self.dec = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, num_outputs*dim_output))
        
        
    def pre_encode(self, X):
        return self.pre_enc(X)
            
            
    def forward(self, X):
        # X: [B, n, x_dim]
        
        # run pre-encoder if neccessary (e.g.: conv encoder for images):
        X = self.pre_encode(X)
        
        data = data.view(data.size(0), data.size(1), -1)
        X = self.enc(X).mean(-2)
        X = self.dec(X).reshape(-1, self.num_outputs, self.dim_output)
        return X


class SetTransformer(nn.Module):
    def __init__(self, params, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super(SetTransformer, self).__init__()
        
        self.pre_enc = get_encoder(params)
        
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
                # SAB(dim_input, dim_hidden, num_heads, ln=ln),
                # SAB(dim_hidden, dim_hidden, num_heads, ln=ln))
        
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),  # PMA returns k * d
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),   # returns k * d
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),   # returns k * d
                nn.Linear(dim_hidden, dim_output))  # returns k * dim_output


    def pre_encode(self, X):
        return self.pre_enc(X)


    def forward(self, X):
        # X: [B, n, D] or [B, n, 28, 28]...
        # run pre-encoder if neccessary (e.g.: conv encoder for images):
        pre_enc_output = self.pre_encode(X)
        X = pre_enc_output.view(pre_enc_output.size(0), pre_enc_output.size(1), -1)
        # print('[fw] X.shape', X.shape)
        out_enc = self.enc(X)  # [B, n, dim_hidden]
        out_dec = self.dec(out_enc)  # [B, num_outputs(=K), dim_output(=2+D)]
        return out_dec, pre_enc_output
