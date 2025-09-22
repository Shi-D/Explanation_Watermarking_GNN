import config
from config import *
import copy
import numpy as np 

from graphlime import GraphLIME

import os
import pickle
import random
import torch
from   torch_geometric.data import Data

from general_utils import *
from models import *
from regression_utils import *
from subgraph_utils import *
import torch.nn.functional as F
from transform_functions import *
from watermark_utils import *
from data_utils import *


def create_random_trigger_graph(num_nodes_trigger, prob_edge, proportion_ones, num_classes, feature_dim, seed):
    np.random.seed(seed)
    trigger_A = np.random.rand(num_nodes_trigger, num_nodes_trigger)<prob_edge
    np.fill_diagonal(trigger_A,0)
    trigger_A = torch.tensor(trigger_A)
    trigger_edge_index = trigger_A.nonzero(as_tuple=True)
    trigger_edge_index = torch.stack(trigger_edge_index, dim=0)
    trigger_X = np.zeros((num_nodes_trigger,feature_dim))
    num_ones = int(feature_dim*proportion_ones)
    for i in range(num_nodes_trigger):
        ones_indices=np.random.choice(feature_dim, num_ones, replace=False)
        trigger_X[i,ones_indices]=1
    trigger_X  =torch.tensor(trigger_X,dtype=torch.float)
    np.random.seed(seed)
    trigger_y=torch.tensor(np.random.randint(0,num_classes, size=num_nodes_trigger),dtype=torch.long)
    trigger_graph = Data(x=trigger_X,edge_index=trigger_edge_index,y=trigger_y)
    return trigger_graph



def backdoor_GraphLIME(dataset_name, dataset, training_indices, attack_target_label,poison_rate=0.1, watermark_size=0.2, seed=0):
    model_folder_config_name_seed_version_name = get_results_folder_name(dataset_name)
    backdoor_items_path = os.path.join(model_folder_config_name_seed_version_name,'graphlime_backdoor_items')
    num_nodes_to_poison = int(poison_rate*len(dataset.x))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random_indices = torch.randperm(len(training_indices))[:num_nodes_to_poison]
    poisoned_node_indices = training_indices[random_indices]
    num_node_features = dataset.x.shape[1]
    num_backdoor_features = int(watermark_size*num_node_features)
    watermark = torch.randint(0, 2, (num_backdoor_features,),dtype=torch.float)

    assert torch.max(poisoned_node_indices)<=torch.max(training_indices)
    if poison_rate==1:
        assert len(poisoned_node_indices) == len(dataset.x)
    print('backdoor_items_path:',backdoor_items_path)
    if os.path.exists(backdoor_items_path)==False or config.watermark_random_backdoor_re_explain==True:
        if config.watermark_random_backdoor_re_explain==False:
            print('GraphLIME explanations haven\'t been generated yet. Doing that now...')
        elif config.watermark_random_backdoor_re_explain==True:
            print('Generating GraphLIME explanations...')
        ranked_feature_dict = get_ranked_features_GraphLIME(dataset_name, dataset, poisoned_node_indices)
        backdoor_info = {'watermark':watermark, 'ranked_feature_dict':ranked_feature_dict}
        for idx in poisoned_node_indices:
            idx = idx.item()
            feature_indices_ranked = ranked_feature_dict[idx]
            backdoor_feature_indices = feature_indices_ranked[:num_backdoor_features]
            dataset.x[idx][backdoor_feature_indices]=watermark
            dataset.y[idx]=attack_target_label
        pickle.dump([dataset, backdoor_info],open(backdoor_items_path,'wb'))
        
    else:
        [dataset, backdoor_info] = pickle.load(open(backdoor_items_path,'rb'))
        ranked_feature_dict = backdoor_info['ranked_feature_dict']
        assert set(poisoned_node_indices.tolist())==set(backdoor_info['ranked_feature_dict'].keys())
        assert torch.all(watermark==backdoor_info['watermark'])
    return dataset, backdoor_info



def get_ranked_features_GraphLIME(dataset_name, data, node_indices):
    config.optimization_kwargs['clf_only']=True
    results_folder = get_results_folder_name(dataset_name)
    clf_only_node_classifier_path = os.path.join(results_folder, 'node_classifier')
    clf_only_node_classifier = pickle.load(open(clf_only_node_classifier_path,'rb'))
    # ranked from least important to most important
    feature_importance_dict = {node_idx.item():None for node_idx in node_indices}
    for i, node_idx in enumerate(node_indices):
        print(f'assessing node {i}/{len(node_indices)}')#,end='\r')
        explainer = GraphLIME(clf_only_node_classifier, hop=1, rho=0.1)
        coefs = explainer.explain_node(node_idx.item(), data.x, data.edge_index)
        abs_coefficients = np.abs(coefs)
        least_representative_indices = np.argsort(abs_coefficients)
        feature_importance_dict[node_idx.item()]=least_representative_indices
    config.optimization_kwargs['clf_only']=False
    return feature_importance_dict



def inject_backdoor_trigger_subgraph(dgl_graph, trigger_graph=None):
    ''' Inserts trigger at random nodes in graph. '''
    num_trigger_nodes = len(trigger_graph.nodes())
    if num_trigger_nodes > len(dgl_graph.nodes()):
        raise ValueError("Number of nodes to select is greater than the number of nodes in the graph.")
    rand_select_nodes=[]
    remaining_nodes = list(dgl_graph.nodes())
    for _ in range(num_trigger_nodes):
        if rand_select_nodes:
            no_edge_nodes = [n for n in remaining_nodes if all(not dgl_graph.has_edges_between(n, m) and not dgl_graph.has_edges_between(m, n) for m in rand_select_nodes)]
            if no_edge_nodes:
                new_node = random.choice(no_edge_nodes)
            else:
                new_node = random.choice(remaining_nodes)
        else:
            new_node = random.choice(remaining_nodes)
        rand_select_nodes.append(new_node.item())
        remaining_nodes.remove(new_node.item())

    node_mapping = {trigger_node: main_node for trigger_node, main_node in zip(trigger_graph.nodes(), rand_select_nodes)}
    edges_start = [[t0.item(), t1.item()] for [t0,t1] in zip(*dgl_graph.edges())]
    edges_final = copy.deepcopy(edges_start)
    ''' Remove any existing connections between selected trigger nodes. '''
    for n0 in rand_select_nodes:
        for n1 in rand_select_nodes:
            if [n0, n1] in edges_final:
                edge_id = dgl_graph.edge_ids(torch.tensor([n0]), torch.tensor([n1]))
                dgl_graph.remove_edges(edge_id)
                edges_final.remove([n0, n1])

    ''' Add edges specified by trigger graph. '''
    trigger_edges = []
    for e in trigger_graph.edges():
        edge = [node_mapping[e[0]], node_mapping[e[1]]]
        (n0, n1) = (min(edge), max(edge))
        trigger_edges.append([n0,n1])
        edges_final.append([n0, n1])
        dgl_graph.add_edges(torch.tensor([n0]), torch.tensor([n1]))
    edges_final = [[t0.item(), t1.item()] for [t0,t1] in zip(*dgl_graph.edges())]
    return dgl_graph, trigger_edges

def inject_backdoor_trigger_features(dgl_graph, trigger_graph=None):
    ''' Inserts trigger at random nodes in graph. '''
    num_trigger_nodes = len(trigger_graph.nodes())
    if num_trigger_nodes > len(dgl_graph.nodes()):
        raise ValueError("Number of nodes to select is greater than the number of nodes in the graph.")
    rand_select_nodes=[]
    remaining_nodes = list(dgl_graph.nodes())
    for _ in range(num_trigger_nodes):
        if rand_select_nodes:
            no_edge_nodes = [n for n in remaining_nodes if all(not dgl_graph.has_edges_between(n, m) and not dgl_graph.has_edges_between(m, n) for m in rand_select_nodes)]
            if no_edge_nodes:
                new_node = random.choice(no_edge_nodes)
            else:
                new_node = random.choice(remaining_nodes)
        else:
            new_node = random.choice(remaining_nodes)
        rand_select_nodes.append(new_node.item())
        remaining_nodes.remove(new_node.item())

    node_mapping = {trigger_node: main_node for trigger_node, main_node in zip(trigger_graph.nodes(), rand_select_nodes)}
    edges_start = [[t0.item(), t1.item()] for [t0,t1] in zip(*dgl_graph.edges())]
    edges_final = copy.deepcopy(edges_start)
    ''' Remove any existing connections between selected trigger nodes. '''
    for n0 in rand_select_nodes:
        for n1 in rand_select_nodes:
            if [n0, n1] in edges_final:
                edge_id = dgl_graph.edge_ids(torch.tensor([n0]), torch.tensor([n1]))
                dgl_graph.remove_edges(edge_id)
                edges_final.remove([n0, n1])

    ''' Add edges specified by trigger graph. '''
    trigger_edges = []
    for e in trigger_graph.edges():
        edge = [node_mapping[e[0]], node_mapping[e[1]]]
        (n0, n1) = (min(edge), max(edge))
        trigger_edges.append([n0,n1])
        edges_final.append([n0, n1])
        dgl_graph.add_edges(torch.tensor([n0]), torch.tensor([n1]))
    edges_final = [[t0.item(), t1.item()] for [t0,t1] in zip(*dgl_graph.edges())]
    return dgl_graph, trigger_edges


def refine_K_and_P(trigger_size, graph_type='ER', K=0, P=0):
    # Chooses automatically if K==0 or P==0, otherwise validates user choice
    '''
    For an Erdos-Renyi (ER) graph generation, P is the probability of adding an edge between any two nodes. If 0, will automatically reset to 1.
    For Small-World (SW) graph generation, P is the probability of rewiring each edge. If 0, will automaticall reset to 1.
    For Small-World (SW) graph generation, K is the number of neighbors in initial ring lattice. If 0, will automatically compute default value as a function of trigger size.
    For Preferential-Attachment (PA) graph generation, K is the number of edges to attach from a new node to existing nodes. If 0, will automatically compute default value as a function of trigger size.
    '''
    def validate_K(graph_type, K, trigger_size):
        if graph_type=='ER':
            return True # K doesn't apply for ER graphs
        elif graph_type == 'PA' and K > trigger_size:
            print('Invalid K: for PA graphs, K must be less than or equal to trigger size.'); return False
        elif graph_type == 'SW' and K<2:
            print('Invalid K: for SW graphs, K must be greater than 2.'); return False
        else:
            return True
    if graph_type=='SW' or graph_type=='PA':
        assert K is not None
        if K==0:
            K=trigger_size=1
            assert validate_K(graph_type, K, trigger_size)
    elif graph_type=='SW' or graph_type=='ER':
        assert P is not None
        if P==0:
            P=1
    return K,P

