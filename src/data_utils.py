import config
import copy
import numpy as np


import os
import pickle
import random
import torch
from   torch_geometric.data import Data
from   torch_geometric.transforms import Compose

from config import *
from general_utils import *
from models import *
from regression_utils import *
from subgraph_utils import *
import torch.nn.functional as F
from transform_functions import *
from watermark_utils import *


def initialize_experiment_data(dataset_name, train_ratio, seed, save_data, load_data):
    """Prepare the dataset based on configuration."""
    val_ratio = test_ratio = (1 - train_ratio) / 2
    split = [train_ratio, val_ratio, test_ratio]
    
    if dataset_attributes[dataset_name]['single_or_multi_graph'] == 'single':
        dataset = prep_data(
            dataset_name=dataset_name,
            location='default',
            batch_size='default',
            transform_list='default',
            train_val_test_split=split,
            seed=seed,
            save=save_data,
            load=load_data
        )
        data = dataset[0]
        data_original = copy.deepcopy(data)
        return dataset, data, data_original
    
    elif dataset_attributes[dataset_name]['single_or_multi_graph'] == 'multi':
        result = prep_data(
            dataset_name=dataset_name,
            location='default',
            batch_size='default',
            transform_list='default',
            train_val_test_split=split
        )
        [train_dataset, val_dataset, test_dataset], [train_loader, val_loader, test_loader] = result
        return result, None, None


def prep_data(dataset_name='CS', location='default', batch_size='default', transform_list = 'default',
              train_val_test_split=[0.6,0.2,0.2], seed=0, load=True, save=False, verbose=True):
    train_ratio, val_ratio, test_ratio = train_val_test_split
    class_ = dataset_attributes[dataset_name]['class']
    single_or_multi_graph = dataset_attributes[dataset_name]['single_or_multi_graph']

    if location=='default':
        location = '../data' if dataset_name in ['PubMed','computers','photo','CS'] else f'../data/{dataset_name}' if dataset_name in ['Reddit2'] else None
    if batch_size=='default':
        batch_size = 'All'
    if transform_list=='default' and dataset_name != 'Reddit2': ### NOTE:  'dataset_name != 'Reddit2' is just a TEMPORARY workaround because i realized something is wrong with CreateMaskTransform and I was working on Reddit2 after I had already collected results for the other datasets.
        transform_list = [CreateMaskTransform(train_ratio, val_ratio, test_ratio, seed)]
    else:
        transform_list = []
    transform = Compose(transform_list)

    if single_or_multi_graph=='single':
        saved_location = f'../data/{dataset_name}/load_this_dataset_trn_{train_val_test_split[0]:.2f}_val_{train_val_test_split[1]:2f}_test_{train_val_test_split[2]:2f}.pkl'
        if load==True:
            try:
                print('Attempting to load dataset from:',saved_location)
                dataset = pickle.load(open(saved_location,'rb'))
                print('Loaded!')
                # if verbose==True:
                return dataset
            except:
                print(f'No saved dataset exists at path:\n{saved_location}')
                print("Existing paths:")
                for f in os.listdir(f'../data/{dataset_name}'):
                    print(f'-- {f}')
                print('\nCreating dataset from scratch.')
                load=False
        if load==False:
            print("LOAD FALSE")
            if dataset_name in ['Reddit', 'Reddit2','Flickr','NELL']:
                dataset = class_(location, transform)
            elif dataset_name in ['RelLinkPredDataset']:
                dataset = class_(location, 'FB15k-237', transform=transform)
            elif dataset_name in ['Twitch_EN']:
                dataset = class_(location, 'EN', transform=transform)
            else:
                dataset = class_(location, dataset_name, transform=transform)
            dataset = add_indices(dataset)
            if save==True:
                print('Dataset being saved at:',saved_location)
                with open(saved_location,'wb') as f:
                    pickle.dump(dataset, f)
        return dataset
    
    elif single_or_multi_graph=='multi':
        saved_train_dataset_location = f'../data/{dataset_name}/load_this_train_dataset_split_amount_{train_val_test_split[0]:.2f}.pkl'
        saved_val_dataset_location = f'../data/{dataset_name}/load_this_val_dataset_split_amount_{train_val_test_split[1]:2f}.pkl'
        saved_test_dataset_location = f'../data/{dataset_name}/load_this_test_dataset_split_amount_{train_val_test_split[2]:2f}.pkl'
        saved_train_loader_location = f'../data/{dataset_name}/load_this_train_loader_split_amount_{train_val_test_split[0]:.2f}.pkl'
        saved_val_loader_location = f'../data/{dataset_name}/load_this_val_loader_split_amount_{train_val_test_split[1]:2f}.pkl'
        saved_test_loader_location = f'../data/{dataset_name}/load_this_test_loader_split_amount_{train_val_test_split[2]:2f}.pkl'
        if load==True:
            try:
                train_dataset = pickle.load(open(saved_train_dataset_location,'rb'))
                val_dataset = pickle.load(open(saved_val_dataset_location,'rb'))
                test_dataset = pickle.load(open(saved_test_dataset_location,'rb'))
                train_loader = pickle.load(open(saved_train_loader_location,'rb'))
                val_loader = pickle.load(open(saved_val_loader_location,'rb'))
                test_loader = pickle.load(open(saved_test_loader_location,'rb'))
            except:
                print('No saved data exists.')
                print("Existing paths:")
                for f in os.listdir(f'../data/{dataset_name}'):
                    print(f'-- {f}')
                print('\nCreating data from scratch.')
                load=False

        if load==False:
            train_dataset = class_(location, split='train', transform=transform)
            train_dataset = add_indices(train_dataset)
            val_dataset   = class_(location, split='val',   transform=transform)
            val_dataset = add_indices(val_dataset)
            test_dataset  = class_(location, split='test',  transform=transform)
            test_dataset = add_indices(test_dataset)

            train_dataset.y = torch.argmax(train_dataset.y, dim=1)
            val_dataset.y   = torch.argmax(val_dataset.y,   dim=1)
            test_dataset.y  = torch.argmax(test_dataset.y,  dim=1)

            batch_size = len(train_dataset) if batch_size=='All' else batch_size

            train_loader = DataLoader(train_dataset,    batch_size=batch_size,  shuffle=True)
            val_loader   = DataLoader(val_dataset,      batch_size=2,           shuffle=False)
            test_loader  = DataLoader(test_dataset,     batch_size=2,           shuffle=False)

            if save==True:
                with open(saved_train_dataset_location,'wb') as f:
                    pickle.dump(train_dataset, f)
                with open(saved_val_dataset_location,'wb') as f:
                    pickle.dump(val_dataset, f)
                with open(saved_test_dataset_location,'wb') as f:
                    pickle.dump(test_dataset, f)
                with open(saved_train_loader_location,'wb') as f:
                    pickle.dump(train_loader, f)
                with open(saved_val_loader_location,'wb') as f:
                    pickle.dump(val_loader, f)
                with open(saved_test_loader_location,'wb') as f:
                    pickle.dump(test_loader, f)
        return [train_dataset, val_dataset, test_dataset], [train_loader, val_loader, test_loader]
    

def prep_data_graph_clf(dataset_name='CS', 
              location='default', 
              batch_size='default',
              transform_list = 'default', 
              train_val_test_split=[0.6,0.2,0.2],
              seed=0,
              load=True,
              save=False,
              verbose=True):

    saved_train_dataset_location = f'../data/{dataset_name}/load_this_train_dataset_split_amount_{train_val_test_split[0]:.2f}.pkl'
    saved_val_dataset_location = f'../data/{dataset_name}/load_this_val_dataset_split_amount_{train_val_test_split[1]:2f}.pkl'
    saved_test_dataset_location = f'../data/{dataset_name}/load_this_test_dataset_split_amount_{train_val_test_split[2]:2f}.pkl'
    saved_train_loader_location = f'../data/{dataset_name}/load_this_train_loader_split_amount_{train_val_test_split[0]:.2f}.pkl'
    saved_val_loader_location = f'../data/{dataset_name}/load_this_val_loader_split_amount_{train_val_test_split[1]:2f}.pkl'
    saved_test_loader_location = f'../data/{dataset_name}/load_this_test_loader_split_amount_{train_val_test_split[2]:2f}.pkl'
    if load==True:
        try:
            train_dataset = pickle.load(open(saved_train_dataset_location,'rb'))
            val_dataset = pickle.load(open(saved_val_dataset_location,'rb'))
            test_dataset = pickle.load(open(saved_test_dataset_location,'rb'))
            train_loader = pickle.load(open(saved_train_loader_location,'rb'))
            val_loader = pickle.load(open(saved_val_loader_location,'rb'))
            test_loader = pickle.load(open(saved_test_loader_location,'rb'))
        except:
            print('No saved data exists.')
            print("Existing paths:")
            for f in os.listdir(f'../data/{dataset_name}'):
                print(f'-- {f}')
            print('\nCreating data from scratch.')
            load=False

    if load==False:

        train_dataset = class_(location, split='train', transform=transform)
        train_dataset = add_indices(train_dataset)
        val_dataset   = class_(location, split='val',   transform=transform)
        val_dataset = add_indices(val_dataset)
        test_dataset  = class_(location, split='test',  transform=transform)
        test_dataset = add_indices(test_dataset)

        train_dataset.y = torch.argmax(train_dataset.y, dim=1)
        val_dataset.y   = torch.argmax(val_dataset.y,   dim=1)
        test_dataset.y  = torch.argmax(test_dataset.y,  dim=1)

        batch_size = len(train_dataset) if batch_size=='All' else batch_size

        train_loader = DataLoader(train_dataset,    batch_size=batch_size,  shuffle=True)
        val_loader   = DataLoader(val_dataset,      batch_size=2,           shuffle=False)
        test_loader  = DataLoader(test_dataset,     batch_size=2,           shuffle=False)

        if save==True:
            with open(saved_train_dataset_location,'wb') as f:
                pickle.dump(train_dataset, f)
            with open(saved_val_dataset_location,'wb') as f:
                pickle.dump(val_dataset, f)
            with open(saved_test_dataset_location,'wb') as f:
                pickle.dump(test_dataset, f)
            with open(saved_train_loader_location,'wb') as f:
                pickle.dump(train_loader, f)
            with open(saved_val_loader_location,'wb') as f:
                pickle.dump(val_loader, f)
            with open(saved_test_loader_location,'wb') as f:
                pickle.dump(test_loader, f)
    return [train_dataset, val_dataset, test_dataset], [train_loader, val_loader, test_loader]
    


def augment_data(
    data, 
    clf_only,
    node_aug, 
    edge_aug, 
    train_nodes_to_consider, 
    all_subgraph_indices, 
    sampling_used=False, 
    seed=0
    ):

    p = config.augment_kwargs['p']
    # def apply_augmentations(data_subset, aug_fns):
    #     for aug_fn in aug_fns:
    #         if aug_fn is not None:
    #             data_subset = aug_fn(data_subset)
    #             print('data_subset:',data_subset)
    #     return data_subset

    def apply_augmentations(data_subset, aug_fns):
        # Save attributes that need to be preserved
        preserved_attrs = {}
        for attr in ['val_mask', 'idx']:
            if hasattr(data_subset, attr):
                preserved_attrs[attr] = getattr(data_subset, attr)
        
        # Apply augmentations
        for aug_fn in aug_fns:
            if aug_fn is not None:
                data_subset = aug_fn(data_subset)
        
        # Restore preserved attributes
        for attr, value in preserved_attrs.items():
            setattr(data_subset, attr, value)
        
        return data_subset
    
    def update_data(data, indices, augmented_data):
        data.x[indices] = augmented_data.x
        data.y[indices] = augmented_data.y
        mask = torch.isin(data.edge_index[0], indices) & torch.isin(data.edge_index[1], indices)
        data.edge_index[:, mask] = augmented_data.edge_index

    def select_random_indices(indices, p):
        n = len(indices)
        torch.manual_seed(seed)
        random_order = torch.randperm(n)
        num_keep = int(p*n)
        keep_indices = indices[random_order[:num_keep]]
        return keep_indices

    augment_subgraphs_separately = config.augment_kwargs['separate_trainset_from_subgraphs'] == True and config.using_our_method==True
    if augment_subgraphs_separately==True:
        if sampling_used==True:
            original_node_indices = data.node_idx
            original_to_new_node_mapping = {original_idx.item():new_idx for (new_idx,original_idx) in zip(range(len(original_node_indices)), original_node_indices)}
            train_nodes_to_consider = torch.tensor([original_to_new_node_mapping[original_idx] for original_idx in train_nodes_to_consider])
            all_subgraph_indices    = torch.tensor([original_to_new_node_mapping[original_idx] for original_idx in all_subgraph_indices])

        trn_minus_subgraph_nodes = torch.tensor(list(set(train_nodes_to_consider.tolist())-set(all_subgraph_indices)))
        trn_minus_subgraph_nodes_keep = select_random_indices(trn_minus_subgraph_nodes, p)
        train_minus_subgraph_data = get_subgraph_from_node_indices(data, trn_minus_subgraph_nodes_keep)
        train_minus_subgraph_data = apply_augmentations(train_minus_subgraph_data, [node_aug, edge_aug])
        update_data(data, trn_minus_subgraph_nodes_keep, train_minus_subgraph_data)
        if config.augment_kwargs['ignore_subgraphs']==False:
            all_subgraph_indices_keep = select_random_indices(all_subgraph_indices,p)
            subgraph_data = get_subgraph_from_node_indices(data, all_subgraph_indices_keep)
            subgraph_data = apply_augmentations(subgraph_data, [node_aug, edge_aug])
            update_data(data, all_subgraph_indices_keep, subgraph_data)
            del subgraph_data
            del all_subgraph_indices_keep
        del trn_minus_subgraph_nodes_keep
        del train_minus_subgraph_data
    elif augment_subgraphs_separately==False or clf_only==True:
        train_nodes = torch.where(data.train_mask==True)[0]
        train_nodes_to_augment = select_random_indices(train_nodes, p)
        train_data_augmented = get_subgraph_from_node_indices(data, train_nodes_to_augment)
        train_minus_subgraph_data = apply_augmentations(train_data_augmented, [node_aug, edge_aug])
        update_data(data, train_nodes_to_augment, train_data_augmented)
        # data = apply_augmentations(data, [node_aug, edge_aug])
    return data 



def create_dataset_from_files(root, dataset_name, sample_size=None):
    '''For graph classification task (MUTAG)'''
    edges = pd.read_csv(os.path.join(root, f"{dataset_name}_A.txt"), header=None, sep=",")
    graph_indicator = pd.read_csv(os.path.join(root, f"{dataset_name}_graph_indicator.txt"), header=None)
    graph_labels = pd.read_csv(os.path.join(root, f"{dataset_name}_graph_labels.txt"), header=None)
    node_labels = pd.read_csv(os.path.join(root, f"{dataset_name}_node_labels.txt"), header=None)

    data_list = []
    N = graph_labels.shape[0] 
    indices_to_use = range(1, N + 1)  

    if sample_size is not None:
        print(f"Taking random sample of size {sample_size}")
        indices_to_use = random.sample(range(1, N + 1), sample_size)

    for c, i in enumerate(indices_to_use):
        print(f"Processing graph {c + 1}/{len(indices_to_use)}", end="\r")
        node_indices = graph_indicator[graph_indicator[0] == i].index
        node_idx_map = {idx: j for j, idx in enumerate(node_indices)}
        graph_edges = edges[edges[0].isin(node_indices + 1) & edges[1].isin(node_indices + 1)]
        graph_edges = graph_edges.apply(lambda col: col.map(lambda x: node_idx_map[x - 1]))
        edge_index = torch.tensor(graph_edges.values, dtype=torch.long).t().contiguous()
        x = torch.tensor(node_labels.iloc[node_indices].values, dtype=torch.float)
        y = torch.tensor(graph_labels.iloc[i - 1].values, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)

    return data_list


def get_subgraph_from_node_indices(data, node_indices):
    sub_edge_index, _ = subgraph(node_indices, data.edge_index, relabel_nodes=False, num_nodes=data.num_nodes)
    sub_data = Data(
        x=data.x[node_indices] if data.x is not None else None,
        edge_index=sub_edge_index,
        y=data.y[node_indices] if data.y is not None else None,
        train_mask=data.train_mask[node_indices] if data.train_mask is not None else None,
        test_mask=data.test_mask[node_indices] if data.test_mask is not None else None,
        val_mask=data.val_mask[node_indices] if data.val_mask is not None else None)
    del sub_edge_index
    return sub_data

def get_classification_train_nodes(data, all_subgraph_indices, sacrifice_method, size_dataset, train_with_test_set=False, clf_only=False):
    if train_with_test_set==False:
        train_mask = data.train_mask
    else:
        train_mask = data.test_mask
    train_nodes_to_use_mask = copy.deepcopy(train_mask)
    if clf_only==True:
        pass
    else:
        if sacrifice_method is not None:
            train_node_indices = torch.arange(size_dataset)[train_mask]
            train_nodes_not_sacrificed = train_node_indices[~torch.isin(train_node_indices, all_subgraph_indices)]
            train_nodes_not_sacrificed_mask = torch.zeros_like(train_mask, dtype=torch.bool)
            train_nodes_not_sacrificed_mask[train_nodes_not_sacrificed] = True
            train_nodes_to_use_mask = train_nodes_not_sacrificed_mask
    return train_nodes_to_use_mask