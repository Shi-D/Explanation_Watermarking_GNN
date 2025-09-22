import numpy as np
import time
from   torch_geometric.datasets import PPI, Reddit, Reddit2, Planetoid, Coauthor, Flickr, Amazon, NELL, RelLinkPredDataset, Twitch


import os
current_dir = os.path.dirname(os.path.abspath(__file__))


root_dir = os.path.dirname(current_dir)
src_dir  = f'{root_dir}/src'
data_dir = f'{root_dir}/data'
results_dir  = f'{root_dir}/training_results'
compare_dicts_dir = f'{root_dir}/compare_dicts'



dataset_attributes = { 
    'CORA': {
        'single_or_multi_graph': 'single',
        'class': Planetoid,
        'num_classes':7,
        'num_nodes':2708,
    },
    'CiteSeer': {
        'single_or_multi_graph': 'single',
        'class': Planetoid,
        'num_classes':6,
        'num_nodes':3327,
    },
    'PubMed': {
        'single_or_multi_graph': 'single',
        'class': Planetoid,
        'num_classes':3,
        'num_nodes':19717,
        'num_features':500,
        'train_ratio':0.6
    },
    'CS': {
        'single_or_multi_graph': 'single',
        'class': Coauthor,
        'num_classes':15,
        'num_nodes':18333,
        'num_features':6805,
        'train_ratio':0.6
    },
    'computers': {
        'single_or_multi_graph': 'single',
        'class': Amazon,
        'num_classes':10,
        'num_nodes':13752,
        'num_features':767,
        'train_ratio':0.6
    },
    'photo': {
        'single_or_multi_graph': 'single',
        'class': Amazon,
        'num_classes':8,
        'num_nodes':7650,
        'num_features':745,
        'train_ratio':0.6
    },
    'MUTAG': {
        'single_or_multi_graph': None,
        'class': None,
        'num_classes': 2,
        'num_nodes':None
    },
    'Reddit2': {
        'single_or_multi_graph': 'single',
        'class': Reddit2,
        'num_classes': 41,
        'num_nodes': 232965,
        'train_ratio':0.6
    }
}




def get_presets(dataset, dataset_name):
    global seed, random_seed, node_classifier_kwargs, optimization_kwargs, \
        using_our_method,\
            watermark_kwargs, subgraph_kwargs, augment_kwargs, watermark_loss_kwargs, regression_kwargs, \
            is_kd_attack, kd_alpha, kd_temp, KD_student_optimization_kwargs,KD_student_node_classifier_kwargs,kd_train_on_subgraphs,kd_subgraphs_only,\
                watermark_random_backdoor, watermark_random_backdoor_prob_edge, watermark_random_backdoor_proportion_ones, watermark_random_backdoor_trigger_size_proportion, watermark_random_backdoor_trigger_alpha, \
                    watermark_graphlime_backdoor, watermark_graphlime_backdoor_target_label, watermark_graphlime_backdoor_poison_rate, watermark_graphlime_backdoor_size,watermark_random_backdoor_re_explain,\
                    preserve_edges_between_subsets


    using_our_method=True

    seed = 0
    random_seed = int(time.time())
    preserve_edges_between_subsets=False

    #########################
    ### other experiments ###
    #########################

    # Knowledge Distillation
    is_kd_attack=False
    kd_alpha=0.5
    kd_temp=1.0
    kd_train_on_subgraphs=False
    kd_subgraphs_only=False


    watermark_graphlime_backdoor = False
    watermark_graphlime_backdoor_target_label = 1
    watermark_graphlime_backdoor_poison_rate = 0.05
    watermark_graphlime_backdoor_size = 0.2

    KD_student_node_classifier_kwargs = {'arch': 'KD_SAGE',  
                                'activation': 'elu',        
                                'numLayers':3,    
                                'hDim':256, 
                                'dropout': 0,    
                                'dropout_subgraphs': 0, 
                                'skip_connections':True,    
                                'heads_1':8,    
                                'heads_2':1,    
                                'inDim': dataset.num_features,  
                                'outDim': dataset.num_classes} 
    
    KD_student_optimization_kwargs     =  {'lr': 0.01,
                                'epochs': 200,
                                'sacrifice_kwargs': {'method':None},
                                'clf_only':False,
                                'coefWmk_kwargs':   {'coefWmk':1},
                                'use_pcgrad':False}
    

    # Watermarking with random backdoor trigger
    watermark_random_backdoor=False
    watermark_random_backdoor_prob_edge = 0.05
    watermark_random_backdoor_proportion_ones = 0.5
    watermark_random_backdoor_trigger_size_proportion = 0.05
    watermark_random_backdoor_trigger_alpha = 0.2
    watermark_random_backdoor_re_explain=False


    #### main settings

    node_classifier_kwargs  =  {'arch': 'SAGE',  
                                'activation': 'elu',        
                                'numLayers':3,    
                                'hDim':256, 
                                'dropout': 0,    
                                'dropout_subgraphs': 0, 
                                'skip_connections':True,    
                                'heads_1':8,    
                                'heads_2':1,    
                                'inDim': dataset.num_features,  
                                'outDim': dataset.num_classes,
                                'use_batching':False,
                                'batch_size': 1000,
                                }

    
    optimization_kwargs     =  {'lr': 0.01,
                                'epochs': 200,
                                'sacrifice_kwargs': {'method':None},
                                'clf_only':False,
                                'coefWmk_kwargs':   {'coefWmk':1},
                                'use_pcgrad':False}

    
    watermark_kwargs        =  {'pGraphs': 1, 
                                'percent_of_features_to_watermark':100,
                                'watermark_type':'most_represented'}
    
    
    subgraph_kwargs         =  {'regenerate': True,
                                'method': 'random',
                                'subgraph_size_as_fraction':0.001,
                                'numSubgraphs': 1,
                                'random_kwargs': {},}
    
    regression_kwargs       =  {'lambda': 0.1}

    watermark_loss_kwargs   =  {'epsilon': 0.001}

    
    augment_kwargs          =  {'separate_trainset_from_subgraphs':True, 
                                'p':1,
                                'ignore_subgraphs':True,
                                'nodeDrop':{'use':True,'p':0.45}, 
                                'nodeMixUp':{'use':True,'lambda':0},  
                                'nodeFeatMask':{'use':True,'p':0.2},    
                                'edgeDrop':{'use':True,'p':0.9}}
    
    if dataset_name=='default':
        pass # above settings are fine



    elif dataset_name=='computers':
        node_classifier_kwargs  =  {'arch': 'GCN', 'activation': 'elu', 'numLayers': 3, 'hDim': 256, 'dropout': 0.1, 'dropout_subgraphs': 0, 'skip_connections': True, 'heads_1': 8, 'heads_2': 1, 'inDim': dataset.num_features, 'outDim': dataset.num_classes, 'use_batching': False, 'batch_size': 1000}
        watermark_kwargs        =  {'pGraphs': 1, 'percent_of_features_to_watermark': 3, 'watermark_type': 'most_represented', }
        watermark_loss_kwargs   =  {'epsilon': 0.01}
        optimization_kwargs     =  {'lr': 0.0008, 'epochs': 350, 'sacrifice_kwargs': {'method': None}, 'clf_only': False, 'coefWmk_kwargs': {'coefWmk': 60,}, 
                                    'use_pcgrad': True}
        subgraph_kwargs         =  {'regenerate': True, 'method': 'random', 'subgraph_size_as_fraction': 0.005, 'numSubgraphs': 7, 'random_kwargs': {}, }
        augment_kwargs          =  {'separate_trainset_from_subgraphs': True, 'p': 0.3, 'ignore_subgraphs': True, 'nodeDrop': {'use': True, 'p': 0.1}, 'nodeMixUp': {'use': True, 'lambda': 5}, 'nodeFeatMask': {'use': False, 'p': 0.2}, 'edgeDrop': {'use': True, 'p': 0.1}}
        augment_kwargs['nodeDrop']['p']=0.8
        augment_kwargs['nodeMixUp']['lambda']=80
        augment_kwargs['edgeDrop']['p']=0.8

    elif dataset_name=='CS':
        node_classifier_kwargs  =  {'arch': 'SAGE', 'activation': 'elu', 'numLayers': 3, 'hDim': 256, 'dropout': 0.1, 'dropout_subgraphs': 0, 'skip_connections': True, 'heads_1': 8, 'heads_2': 1, 'inDim': dataset.num_features, 'outDim': dataset.num_classes, 'use_batching': False, 'batch_size': 1000}
        KD_student_node_classifier_kwargs  =  {'arch': 'SAGE', 'activation': 'elu', 'numLayers': 3, 'hDim': 256, 'dropout': 0.1, 'dropout_subgraphs': 0, 'skip_connections': True, 'heads_1': 8, 'heads_2': 1, 'inDim': dataset.num_features, 'outDim': dataset.num_classes}
        watermark_kwargs        =  {'pGraphs': 1, 'percent_of_features_to_watermark': 3, 'watermark_type': 'most_represented', }
        watermark_loss_kwargs   =  {'epsilon': 0.1}
        optimization_kwargs     =  {'lr': 0.001, 'epochs': 90, 'sacrifice_kwargs': {'method': None}, 'clf_only': False, 'coefWmk_kwargs': {'coefWmk': 20, }, 
                                    'use_pcgrad': True}
        KD_student_optimization_kwargs     =  {'lr': 0.001, 'epochs': 90, 'sacrifice_kwargs': {'method': None}, 'clf_only': False, 'coefWmk_kwargs': {'coefWmk': 20, }, 
                                               'use_pcgrad': True}
        subgraph_kwargs         =  {'regenerate': True, 'method': 'random', 'subgraph_size_as_fraction': 0.005, 'numSubgraphs': 7, 'random_kwargs': {}, }
        augment_kwargs          =  {'separate_trainset_from_subgraphs': True, 'p': 0.3, 'ignore_subgraphs': True, 'nodeDrop': {'use': True, 'p': 0.1}, 'nodeMixUp': {'use': True, 'lambda': 5}, 'nodeFeatMask': {'use': False, 'p': 0.2}, 'edgeDrop': {'use': True, 'p': 0.1}}

    elif dataset_name=='photo':
        node_classifier_kwargs =  {'arch': 'GCN', 'activation': 'elu', 'numLayers':3, 'hDim': 256, 'dropout': 0.1, 'dropout_subgraphs': 0, 'skip_connections': True, 'heads_1': 8, 'heads_2': 1, 'inDim': dataset.num_features, 'outDim': dataset.num_classes, 'use_batching': False, 'batch_size': 1000}
        KD_student_node_classifier_kwargs =  {'arch': 'GCN', 'activation': 'elu', 'numLayers':3, 'hDim': 256, 'dropout': 0.1, 'dropout_subgraphs': 0, 'skip_connections': True, 'heads_1': 8, 'heads_2': 1, 'inDim': dataset.num_features, 'outDim': dataset.num_classes}
        watermark_kwargs       =  {'pGraphs': 1, 'percent_of_features_to_watermark': 3, 'watermark_type': 'most_represented', }
        watermark_loss_kwargs  =  {'epsilon': 0.01}
        optimization_kwargs    =  {'lr': 0.0002, 'epochs': 300, 'sacrifice_kwargs': {'method': None}, 'clf_only': False, 'coefWmk_kwargs': {'coefWmk': 100}, 'use_pcgrad': True}
        KD_student_optimization_kwargs    =  {'lr': 0.0002, 'epochs': 300,'sacrifice_kwargs': {'method': None}, 'clf_only': False, 'coefWmk_kwargs': {'coefWmk': 100}, 'use_pcgrad': True}
        subgraph_kwargs        =  {'regenerate': True, 'method': 'random', 'subgraph_size_as_fraction': 0.005, 'numSubgraphs': 7, 'random_kwargs': {}, }
        augment_kwargs         =  {'nodeDrop': {'use': True, 'p': 0.3}, 'nodeMixUp': {'use': True, 'lambda': 1}, 'nodeFeatMask': {'use': True, 'p': 0.3}, 'edgeDrop': {'use': True, 'p': 0.3}, 'p': 0.3, 'separate_trainset_from_subgraphs': True, 'ignore_subgraphs': True}

    elif dataset_name=='PubMed':
        node_classifier_kwargs =  {'arch': 'SAGE', 'activation': 'elu', 'numLayers': 3, 'hDim': 256, 'dropout': 0.1, 'dropout_subgraphs': 0, 'skip_connections': True, 'heads_1': 8, 'heads_2': 1, 'inDim': dataset.num_features, 'outDim': dataset.num_classes, 'use_batching': False, 'batch_size': 1000}
        KD_student_node_classifier_kwargs =  {'arch': 'SAGE', 'activation': 'elu', 'numLayers': 3, 'hDim': 256, 'dropout': 0.1, 'dropout_subgraphs': 0, 'skip_connections': True, 'heads_1': 8, 'heads_2': 1, 'inDim': dataset.num_features, 'outDim': dataset.num_classes}
        watermark_kwargs       =  {'pGraphs': 1, 'percent_of_features_to_watermark': 3, 'watermark_type': 'most_represented'}
        watermark_loss_kwargs  =  {'epsilon': 0.1}
        optimization_kwargs    =  {'lr': 0.005, 'epochs': 200,  'sacrifice_kwargs': {'method': None}, 'clf_only': False, 'coefWmk_kwargs': {'coefWmk': 70, }, 'use_pcgrad': True}
        KD_student_optimization_kwargs    =  {'lr': 0.005, 'epochs': 200,  'sacrifice_kwargs': {'method': None}, 'clf_only': False, 'coefWmk_kwargs': {'coefWmk': 70}, 'use_pcgrad': True}
        subgraph_kwargs        =  {'regenerate': True, 'method': 'random', 'subgraph_size_as_fraction': 0.005, 'numSubgraphs': 7, 'random_kwargs': {}}
        augment_kwargs         =  {'separate_trainset_from_subgraphs': True, 'p': 0, 'ignore_subgraphs': True, 'nodeDrop': {'use': True, 'p': 0.1}, 'nodeMixUp': {'use': True, 'lambda': 5}, 'nodeFeatMask': {'use': False, 'p': 0.2}, 'edgeDrop': {'use': True, 'p': 0.1}}


    elif dataset_name=='Reddit2':
        node_classifier_kwargs =  {'arch': 'GCN', 'activation': 'relu', 'numLayers': 2, 'hDim': 256, 'dropout': 0.5, 'dropout_subgraphs': 0, 'skip_connections': False, 'heads_1': 8, 'heads_2': 1, 'inDim': dataset.num_features, 'outDim': dataset.num_classes,'use_batching':True, 'batch_size':1024}
        augment_kwargs         =  {'separate_trainset_from_subgraphs': True, 'p': 0, 'ignore_subgraphs': True, 'nodeDrop': {'use': False, 'p': 0}, 'nodeMixUp': {'use': False, 'lambda': 0}, 'nodeFeatMask': {'use': False, 'p': 0}, 'edgeDrop': {'use': False, 'p': 0}}
        optimization_kwargs    =  {'lr': 0.01, 'epochs': 200,'sacrifice_kwargs': {'method':None},'clf_only':False,'coefWmk_kwargs':   {'coefWmk':1},'use_pcgrad':False}


def validate_regression_kwargs():
    assert set(list(regression_kwargs.keys()))=={'lambda'}
    assert isinstance(regression_kwargs['lambda'],(int,float,np.integer,np.floating))
    assert regression_kwargs['lambda']>=0


def validate_optimization_kwargs():
    print('optimization keys:',optimization_kwargs.keys())
    assert set(list(optimization_kwargs.keys()))=={'lr','epochs',
                                                   'sacrifice_kwargs','coefWmk_kwargs','clf_only','use_pcgrad'}
    assert isinstance(optimization_kwargs['lr'],(int, float, np.integer, np.floating)) and optimization_kwargs['lr']>=0
    assert isinstance(optimization_kwargs['epochs'],int) and optimization_kwargs['epochs']>=0
    assert isinstance(optimization_kwargs['sacrifice_kwargs'],dict)
    assert set(list(optimization_kwargs['sacrifice_kwargs'].keys()))=={'method'}
    assert optimization_kwargs['sacrifice_kwargs']['method'] in [None,'subgraph_node_indices','train_node_indices']
    assert isinstance(optimization_kwargs['coefWmk_kwargs'],dict)
    assert set(list(optimization_kwargs['coefWmk_kwargs'].keys()))=={'coefWmk'}
    assert isinstance(optimization_kwargs['coefWmk_kwargs']['coefWmk'],(int, float, np.integer, np.floating)) and optimization_kwargs['coefWmk_kwargs']['coefWmk']>=0
    assert isinstance(optimization_kwargs['clf_only'], bool)
    assert isinstance(optimization_kwargs['use_pcgrad'],bool)

def validate_node_classifier_kwargs():
    assert set(list(node_classifier_kwargs.keys()))=={'arch','activation','numLayers','hDim','dropout','dropout_subgraphs','skip_connections','heads_1','heads_2','inDim','outDim','use_batching','batch_size'}
    assert node_classifier_kwargs['arch'] in ['GAT','GCN','GraphConv','SAGE','SGC','Transformer','SAGE_efficient']
    assert isinstance(node_classifier_kwargs['numLayers'],int)
    assert isinstance(node_classifier_kwargs['inDim'],int)
    assert isinstance(node_classifier_kwargs['hDim'],int)
    assert isinstance(node_classifier_kwargs['outDim'],int)
    assert node_classifier_kwargs['dropout']>=0 and node_classifier_kwargs['dropout']<=1
    assert node_classifier_kwargs['dropout_subgraphs']>=0 and node_classifier_kwargs['dropout_subgraphs']<=1
    assert isinstance(node_classifier_kwargs['skip_connections'],bool)
    assert isinstance(node_classifier_kwargs['use_batching'],bool)
    assert isinstance(node_classifier_kwargs['batch_size'],int)

def validate_subgraph_kwargs():
    assert set(list(subgraph_kwargs.keys()))=={'regenerate','method','numSubgraphs','subgraph_size_as_fraction','random_kwargs'}
    assert isinstance(subgraph_kwargs['regenerate'],bool)
    assert isinstance(subgraph_kwargs['numSubgraphs'],int)
    assert isinstance(subgraph_kwargs['subgraph_size_as_fraction'], (int, float, np.integer, np.floating))
    assert subgraph_kwargs['subgraph_size_as_fraction']>0 and subgraph_kwargs['subgraph_size_as_fraction']<1
    assert subgraph_kwargs['method'] in ['random']

def validate_augment_kwargs():
    assert set(list(augment_kwargs.keys()))=={'separate_trainset_from_subgraphs', 'p','ignore_subgraphs','nodeDrop', 'nodeFeatMask','edgeDrop','nodeMixUp'}
    assert isinstance(augment_kwargs['separate_trainset_from_subgraphs'],bool)
    assert isinstance(augment_kwargs['p'],(int, float, np.integer, np.floating)) and augment_kwargs['p']>=0 and augment_kwargs['p']<=1
    assert isinstance(augment_kwargs['ignore_subgraphs'],bool)
    for k in ['nodeDrop', 'nodeFeatMask','edgeDrop','nodeMixUp']:
        assert isinstance(augment_kwargs[k]['use'],bool)
        if k in ['nodeDrop','nodeFeatMask','edgeDrop']:
            assert set(list(augment_kwargs[k].keys()))=={'use','p'}
            assert augment_kwargs[k]['p'] >= 0 and augment_kwargs[k]['p'] <= 1 and isinstance(augment_kwargs[k]['p'], (int, float, np.integer, np.floating))
        elif k=='nodeMixUp':
            assert set(list(augment_kwargs[k].keys()))=={'use','lambda'}
            assert isinstance(augment_kwargs[k]['lambda'],int) or isinstance(augment_kwargs[k]['lambda'],float)

def validate_watermark_kwargs():
    assert set(list(watermark_kwargs.keys()))=={'pGraphs', 'percent_of_features_to_watermark','watermark_type'}
    assert watermark_kwargs['percent_of_features_to_watermark'] >= 0 and watermark_kwargs['percent_of_features_to_watermark'] <= 100 and isinstance(watermark_kwargs['percent_of_features_to_watermark'], (int, float, complex, np.integer, np.floating))
    assert watermark_kwargs['watermark_type'] in ['most_represented']

def validate_watermark_loss_kwargs():
    assert set(list(watermark_loss_kwargs.keys()))=={'epsilon'}
    assert isinstance(watermark_loss_kwargs['epsilon'],(int, float, np.integer, np.floating))
    assert watermark_loss_kwargs['epsilon']>=0

def validate_kwargs():
    assert isinstance(seed,int)
    validate_regression_kwargs()
    validate_optimization_kwargs()
    validate_node_classifier_kwargs()
    validate_subgraph_kwargs()
    validate_augment_kwargs()
    validate_watermark_kwargs()
    validate_watermark_loss_kwargs()
