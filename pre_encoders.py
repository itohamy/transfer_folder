import torch.nn as nn
import torch
import torch.nn.functional as F
from resnet import resnet18, resnet34


def get_encoder(params):
    encoder = None
        
    if params['pre_encoder_type'] == 'conv':
        encoder = conv_encoder(params)
    elif params['pre_encoder_type'] == 'identity':
        encoder = nn.Identity()
    elif params['pre_encoder_type'] == 'resnet18':
        encoder = resnet18(params)   
    elif params['pre_encoder_type'] == 'resnet34':
        encoder = resnet34(params)
    else:
        raise NameError('Unknown pre encoder type ' + params['pre_encoder_type'])
        
    return encoder


class conv_encoder(nn.Module):
    
    def __init__(self, params):  
        super(conv_encoder, self).__init__()
        
        fc_dim = 320
        if params['img_sz'] == 32:
            fc_dim = 500
            
        self.conv1 = nn.Conv2d(params['channels'], 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(fc_dim, 256)
        self.fc2 = nn.Linear(256, params['h_dim'])

    def forward(self, x):
        ''' 
            x: [B, N, x_dim] 
            out: [B, N, h_dim]
        '''
        
        B = x.shape[0]
        N = x.shape[1]
        x = x.reshape(tuple((-1,)) + tuple(x.shape[2:])) # will be: [B*N, channels, img_sz, img_sz] or [B*N, img_sz, img_sz] or [B*N, x_dim]
 
        if len(x.shape) < 4:
            x = x.unsqueeze(1)   # add channel index
        
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)  # [B*N, h_dim]  
        out = x.reshape([B, N, -1])  # [B, N, h_dim]  
        return out


