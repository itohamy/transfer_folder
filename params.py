

def get_parameters(dataset_name):

    params = {}
    
    params['batch_size'] = 64
    params['max_it'] = 50000
    params['data_path'] = '/home/tohamy/Projects/data'  # options: [/home/tohamy/Projects/data, /vildata/tohamy/CPAB_Activation/data'
    params['net_type'] = 'set_transformer'  # Choose from: ['set_transformer', 'deepset']
    
    params['plot_freq'] = 100  # -1, 100..
    params['iter_stats_avg'] = 1000  # from this iteration we start computing stats average (NMI, ARI, LL)
    
    # Optimation params:    
    params['lr'] = 0.0005
    
    params['CIFAR100_TRAIN_MEAN'] = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)  # for data transform
    params['CIFAR100_TRAIN_STD'] = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)   # for data transform

    params['alpha'] = .7  # Dispersion parameter of the Chinese Restaurant Process

    ''' Options for encoder types: [identity, conv, resnet18, resnet34]
        Required params for each encoder:
            conv / resnet*: h_dim
        (if no encoder is used, h_dim is the dimension of the data: [B, n, h_dim])
    '''
    
    if dataset_name == 'Gauss2D':         
        params['sigma'] = 1        # std for the Gaussian noise around the cluster mean 
        params['lambda'] = 10      # std for the Gaussian prior that generates de centers of the clusters
        params['img_sz'] = None
        params['channels'] = 0
        params['nlabels'] = None
        params['K_fixed'] = 6
        params['Nmin'] = 100
        params['Nmax'] = 500
        params['Nmin_test'] = 200
        params['Nmax_test'] = 201
        params['pre_encoder_type'] = 'identity'
        params['h_dim'] = 2
        params['x_dim'] = 2
        
    elif dataset_name == 'MNIST':
        params['img_sz'] = 28
        params['channels'] = 1
        params['nlabels'] = 10
        params['K_fixed'] = 6
        params['Nmin'] = 100
        params['Nmax'] = 500
        params['Nmin_test'] = 200
        params['Nmax_test'] = 201
        params['pre_encoder_type'] = 'conv'
        params['h_dim'] = 256  # 256 / 784  # should be 784 if no pre_encoder is used.  
        params['x_dim'] = None  # used only when channels == 0
        
    elif dataset_name == 'FASHIONMNIST':
        params['img_sz'] = 28
        params['channels'] = 1
        params['nlabels'] = 10
        params['K_fixed'] = 6
        params['Nmin'] = 100
        params['Nmax'] = 1000
        params['Nmin_test'] = 200
        params['Nmax_test'] = 201
        params['pre_encoder_type'] = 'conv'
        params['h_dim'] = 256  # should be 784 if no pre_encoder is used.  
        params['x_dim'] = None  # used only when channels == 0
                
    elif dataset_name == 'CIFAR':
        params['img_sz'] = 32
        params['channels'] = 3
        params['input_dim'] = 32 * 32 * 3
        params['nlabels'] = 10
        params['K_fixed'] = 6
        params['reduce_dim_type'] = 't-SNE'
        params['data_path'] = params['data_path'] + '/CIFAR-10-images/' 
        params['Nmin'] = 100
        params['Nmax'] = 1000
        params['Nmin_test'] = 20
        params['Nmax_test'] = 50
        params['pre_encoder_type'] = 'conv'
        params['h_dim'] = 256
        params['x_dim'] = None  # used only when channels == 0
                
    elif dataset_name == 'Features':
        params['img_sz'] = None
        params['channels'] = 0
        params['nlabels'] = 50
        params['K_fixed'] = 10
        params['data_path'] = params['data_path'] + '/imagenet50_featutres/'  
        params['Nmin'] = 100
        params['Nmax'] = 1300
        params['Nmin_test'] = 20
        params['Nmax_test'] = 50
        params['pre_encoder_type'] = 'identity'
        params['h_dim'] = 384
        params['x_dim'] = 384
                
    elif dataset_name == 'tinyimagenet':
        params['img_sz'] = 64
        params['channels'] = 3
        params['input_dim'] = 64 * 64 * 3
        params['nlabels'] = 200
        params['K_fixed'] = 10
        params['reduce_dim_type'] = 't-SNE'
        params['data_path'] = params['data_path'] + '/tiny_imagenet/tiny-imagenet-200/'
        params['Nmin'] = 50
        params['Nmax'] = 500
        params['Nmin_test'] = 20
        params['Nmax_test'] = 50
        params['pre_encoder_type'] = 'conv'
        params['h_dim'] = 256
        params['x_dim'] = None  # used only when channels == 0
                
    else:
        raise NameError('Unknown dataset_name: '+ dataset_name)
        
    return params