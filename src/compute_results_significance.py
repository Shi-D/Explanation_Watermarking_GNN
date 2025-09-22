from embed_and_verify import *
import argparse
import os
import random
import config
import numpy as np
from   scipy import stats

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

data_dir = os.path.join(parent_dir, 'data')
subgraph_dir = os.path.join(data_dir,'random_subgraphs')


try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except:
    current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Gathering information about natural distributions of matching indices across regression results')
    parser.add_argument('--dataset_name', type=str, default='computers',help='Dataset Name')
    parser.add_argument('--subgraph_size_as_fraction', type=float,  default=0.005, help='List of values representing subgraph size (as fraction of the training data).')

    args, _ = parser.parse_known_args()

    ### load data
    dataset_name = args.dataset_name
    train_ratio = dataset_attributes[dataset_name]['train_ratio']
    val_ratio = test_ratio = (1-train_ratio)/2
    saved_data_filename = f'load_this_dataset_trn_{train_ratio:.2f}_val_{val_ratio:2f}_test_{test_ratio:2f}.pkl'
    data_path = os.path.join(parent_dir, 'data', dataset_name, saved_data_filename)
    dataset = pickle.load(open(data_path,'rb'))
    
    data = dataset[0]

    get_presets(dataset,dataset_name)
    parser.add_argument('--numSubgraphs', type=int, default=config.subgraph_kwargs['numSubgraphs'], help='corresponds to number of subgraphs model was trained on')
    parser.add_argument('--numLayers',type=int,default=config.node_classifier_kwargs['numLayers'],help='Number of layers associated with model architecture')
    parser.add_argument('--hDim',type=int,default=config.node_classifier_kwargs['hDim'],help='Number of hidden features associated with model architecture')
    parser.add_argument('--epsilon',type=float,default=config.watermark_loss_kwargs['epsilon'],help='wmk loss epsilon')    
    parser.add_argument('--coefWmk',type=float,default=config.optimization_kwargs['coefWmk_kwargs']['coefWmk'],help='coef wmk loss')
    parser.add_argument('--dropout',type=float,default=config.node_classifier_kwargs['dropout'],help='GNN dropout rate')
    parser.add_argument('--epochs',type=int,default=config.optimization_kwargs['epochs'],help='number of epochs')
    parser.add_argument('--lr',type=float,default=config.optimization_kwargs['lr'],help='learning rate')
    parser.add_argument('--arch',type=str,default=config.node_classifier_kwargs['arch'],help='Model architecture')

    args = parser.parse_args()

    if args.numSubgraphs==None:
        args.numSubgraphs = config.subgraph_kwargs['numSubgraphs']
    else:
        config.subgraph_kwargs['numSubgraphs'] = args.numSubgraphs
    if args.numLayers==None:
        args.numLayers = config.subgraph_kwargs['numSubgraphs']
    else:
        config.node_classifier_kwargs['numLayers'] = args.numLayers
    if args.hDim==None:
        args.hDim = config.node_classifier_kwargs['numLayers']
    else:
        config.node_classifier_kwargs['hDim'] = args.hDim
    if args.epsilon==None:
        args.epsilon = config.node_classifier_kwargs['hDim']
    else:
        config.watermark_loss_kwargs['epsilon'] = args.epsilon
    if args.coefWmk==None:
        args.coefWmk = config.watermark_loss_kwargs['epsilon']
    else:
        config.optimization_kwargs['coefWmk_kwargs']['coefWmk'] = args.coefWmk
    if args.dropout==None:
        args.dropout = config.optimization_kwargs['coefWmk_kwargs']['coefWmk']
    else:
        config.node_classifier_kwargs['dropout'] = args.dropout
    if args.epochs==None:
        args.epoch = config.optimization_kwargs['epochs']
    else:
        config.optimization_kwargs['epochs'] = args.epochs
    if args.lr==None:
        args.lr = config.optimization_kwargs['lr']
    else:
        config.optimization_kwargs['lr'] = args.lr
    if args.arch==None:
        args.arch = config.node_classifier_kwargs['arch']
    else:
        config.node_classifier_kwargs['arch'] = args.arch




    ###
    def get_numSubgraphs_from_folder_name(folder_name):
        return int(folder_name.split('numSubgraphs')[1].split('_')[0])
    def get_numLayers_from_folder_name(folder_name):
        return int(folder_name.split('numLayers')[1].split('_')[0])
    def get_hDim_from_folder_name(folder_name):
        return int(folder_name.split('hDim')[1].split('_')[0])
    def get_eps_from_folder_name(folder_name):
        return float(folder_name.split('eps')[1].split('_')[0])
    def get_coefWmk_from_folder_name(folder_name):
        return float(folder_name.split('coefWmk')[1].split('_')[0])
    def get_dropout_from_folder_name(folder_name):
        return float(folder_name.split('drop')[1].split('_')[0])
    def get_epochs_from_folder_name(folder_name):
        return int(folder_name.split('epochs')[1].split('_')[0])
    def get_lr_from_folder_name(folder_name):
        return float(folder_name.split('lr')[1].split('_')[0])
    def get_fraction_from_folder_name(folder_name):
        return float(folder_name.split('fraction')[1].split('_')[0])
    def get_arch_from_folder_name(folder_name):
        return folder_name.split('arch')[1].split('_')[0]
    
    subgraph_size_as_fraction = args.subgraph_size_as_fraction
    train_folder = os.path.join(parent_dir, 'training_results',dataset_name)

    arch_folder = []
    for f in os.listdir(train_folder):
        # print('f:',f)
        if len(f)>3 and f[:4]=='arch':
            c1 = get_arch_from_folder_name(f)==args.arch
            c2 = get_numLayers_from_folder_name(f)==args.numLayers
            c3 = get_numLayers_from_folder_name(f)==args.numLayers
            if c1+c2+c3==3:
                arch_folder.append(f)
    arch_folder = arch_folder[0]
    arch_folder = os.path.join(train_folder, arch_folder)
    model_paths = [os.path.join(arch_folder,f) for f in os.listdir(arch_folder) if f[0]!='.' and 'ignore' not in f]

    try:
        eligible_model_folders = [f for f in model_paths if \
                                                    get_numSubgraphs_from_folder_name(f)==args.numSubgraphs and \
                                                    get_eps_from_folder_name(f)==args.epsilon and \
                                                    int(get_coefWmk_from_folder_name(f))==args.coefWmk and \
                                                    get_dropout_from_folder_name(f)==args.dropout and \
                                                    get_epochs_from_folder_name(f)==args.epochs and \
                                                    get_lr_from_folder_name(f)==args.lr and \
                                                    get_fraction_from_folder_name(f)==subgraph_size_as_fraction #and \ 
                                                    ]
        model_folder = eligible_model_folders[0]
    except:
        print('****')
        for m in model_paths:
            print(f'\n\n{m}\n')
            print('args.numSubgraphs==get_numSubgraphs_from_folder_name(f):',args.numSubgraphs,get_numSubgraphs_from_folder_name(m))
            print('args.numLayers==get_numLayers_from_folder_name(f):',args.numLayers,get_numLayers_from_folder_name(m))
            print('args.hDim==get_hDim_from_folder_name(f):',args.hDim,get_hDim_from_folder_name(m))
            print('args.epsilon==get_eps_from_folder_name(f):',args.epsilon,get_eps_from_folder_name(m))
            print('args.coefWmk==get_coefWmk_from_folder_name(f):',args.coefWmk,get_coefWmk_from_folder_name(m))
            print('args.dropout==get_dropout_from_folder_name(f):',args.dropout,get_dropout_from_folder_name(m))
            print('args.epochs==get_epochs_from_folder_name(f):',args.epochs,get_epochs_from_folder_name(m))
            print('args.lr==get_lr_from_folder_name(f):',args.lr,get_lr_from_folder_name(m))
            print('args.subgraph_size_as_fraction==get_fraction_from_folder_name(f):',subgraph_size_as_fraction,get_fraction_from_folder_name(m))

    # model_folder = get_results_folder_name(dataset_name)
    # seed_folders = [os.listdir(m) for m in eligible_model_folders]
    seed_folders = [os.path.join(model_folder,f) for f in os.listdir(model_folder) if f[0]!='.' and 'ignore' not in f and 'seed' in f]


    all_zs = []
    all_ps = []
    for f_name in seed_folders:
        seed = f_name.split('seed')[1]
        seed_folder = os.path.join(arch_folder, model_folder, f_name)
        path_actual_distribution = os.path.join(seed_folder, f'distribution.txt')
        path_training_results = os.path.join(seed_folder, 'results.txt')
        
        try:
            with open(path_training_results,'r') as f:
                training_results_text = f.read()
            training_matches = int(training_results_text.split(f'Seed {seed}')[1].split('#_match_WMK w/wout 0s = ')[1].split(',')[0].split('/')[1])
        except:
            print('training path exists:',os.path.exists(path_training_results))
            print('Could not obtain training results from file. Check to see if you actually trained on this configuration.')

        try:
            with open(path_actual_distribution,'r') as f:
                random_subgraph_distribution_text = f.read()
            random_subgraph_matches_mu = float(random_subgraph_distribution_text.split('mu_natural=')[1].split(',')[0])
            random_subgraph_matches_sigma = float(random_subgraph_distribution_text.split('sigma_natural=')[1])
        except:
            print('distribution path exists:',os.path.exists(path_actual_distribution))
            print('Could not obtain sample distribution info. Check to see if you actually ran collect_random_subgraphs.py on this configuration.')

        
        z_score = (training_matches - random_subgraph_matches_mu)/random_subgraph_matches_sigma
        all_zs.append(z_score)
        p_score = 1 - stats.norm.cdf(z_score)
        all_ps.append(p_score)

        print(f'\nSignificance of results obtained during training, {f_name}:\n')
        print(f'Z-score: {z_score}')
        print(f'P-score: {p_score}')

        with open(os.path.join(seed_folder,'results_actual_significance.txt'),'w') as f:
            f.write('Significance of results obtained during training, using actual sample distribution (rather than estimated distribution):\n')
            f.write(f'Z-score: {z_score}')
            f.write(f'P-score: {p_score}')
    
    with open(os.path.join(model_folder, 'results_actual_significance.txt'),'w') as f:
        f.write('Significance of results obtained during training, using actual sample distribution (rather than estimated distribution), averaged across seeds:\n')
        f.write(f'Z-scores: {np.mean(all_zs)}')
        f.write(f'P-scores: {np.mean(all_ps)}')



    
        