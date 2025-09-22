from   config import *
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv, GCNConv, GraphConv, SAGEConv
from torch_geometric.nn.conv import SGConv, TransformerConv
from torch_geometric.loader import NeighborLoader


from torch_geometric.utils import k_hop_subgraph

class Net(torch.nn.Module):
    def __init__(self, **model_kwargs):
        super(Net, self).__init__()
        inDim = model_kwargs['inDim'] 
        hDim = model_kwargs['hDim']
        self.outDim = model_kwargs['outDim']
        self.dropout = model_kwargs['dropout']
        self.activation_fn = {'elu': F.elu, 'relu': F.relu}[model_kwargs['activation']]
        self.numLayers = model_kwargs['numLayers']
        self.arch = model_kwargs['arch']


        conv_fn = {'GAT': GATConv, 
                   'GCN': GCNConv, 
                   'GraphConv': GraphConv, 
                   'SAGE': SAGEConv, 
                   'SGC': SGConv, 
                   'Transformer': TransformerConv,
                   }[model_kwargs['arch']]
        self.convs = nn.ModuleList()
        if model_kwargs['arch']=='GAT':
            heads_1 = model_kwargs['heads_1']
            heads_2 = model_kwargs['heads_2']
            self.convs.append(conv_fn(in_channels=inDim, out_channels=hDim // heads_1, heads=heads_1, concat=True, dropout=self.dropout)) # First conv layer
            for l in range(self.numLayers - 2): # Intermediate conv layers
                self.convs.append(conv_fn(in_channels=hDim, out_channels=hDim // heads_2, heads=heads_2, concat=True, dropout=self.dropout))
            self.convs.append(conv_fn(in_channels=hDim, out_channels=self.outDim, heads=heads_2, concat=False, dropout=self.dropout)) # Final conv layer
            
        else:
            self.convs.append(conv_fn(in_channels=inDim, out_channels=hDim)) # First conv layer
            for l in range(self.numLayers - 2): # Intermediate conv layers
                self.convs.append(conv_fn(in_channels=hDim, out_channels=hDim))
            self.convs.append(conv_fn(in_channels=hDim, out_channels=self.outDim)) # Final conv layer
        self.skip_connections = model_kwargs['skip_connections']
        if self.skip_connections:
            self.lin = torch.nn.Linear(((self.numLayers-1)*hDim)+self.outDim, self.outDim)
        self.feature_weights = torch.zeros(inDim)  # To track feature weights

        
    def forward(self, x, edge_index, dropout=0):
        intermediate_outputs = []
        for l in range(self.numLayers):
            x = self.convs[l](x,edge_index)

            x = self.activation_fn(x)
            x = F.dropout(x, p=dropout, training=self.training)
            intermediate_outputs.append(x)
        if self.skip_connections == True:
            x = torch.cat(intermediate_outputs, dim=-1)
            x = self.lin(x)
        return F.log_softmax(x, dim=1)

    def update_feature_weights(self, x):
        grad = x.grad.abs().mean(dim=0)
        self.feature_weights += grad

####


class DataLoaderRegistry:
    def __init__(self):
        # Store configuration instead of actual loaders
        self.loader_configs = {}
        # Store current active loaders (not pickled)
        self._active_loaders = {}
    
    def __getstate__(self):
        # Only pickle the configs, not the loaders
        return {'loader_configs': self.loader_configs}
    
    def __setstate__(self, state):
        # When unpickling, restore configs but initialize empty active loaders
        self.loader_configs = state['loader_configs']
        self._active_loaders = {}
    
    def get_loader(self, data, mask, batch_size, mode, **kwargs):
        # Create a unique key for this loader configuration
        key = f"{id(data)}_{id(mask)}_{batch_size}_{mode}"
        
        # Store config if it doesn't exist
        if key not in self.loader_configs:
            if mode == 'train':
                num_neighbors = [25, 10]
                shuffle = True
            else:
                num_neighbors = [-1]
                shuffle = False
            
            self.loader_configs[key] = {
                'mask': mask,
                'batch_size': batch_size,
                'num_neighbors': num_neighbors,
                'shuffle': shuffle,
                'kwargs': kwargs
            }
        
        # Create or return the active loader
        if key not in self._active_loaders:
            config = self.loader_configs[key]
            from torch_geometric.loader import NeighborLoader
            loader_kwargs = {'batch_size': config['batch_size'], 'num_workers': 6, 'persistent_workers': True}
            loader_kwargs.update(config.get('kwargs', {}))
            
            self._active_loaders[key] = NeighborLoader(
                data, 
                input_nodes=config['mask'],
                num_neighbors=config['num_neighbors'], 
                shuffle=config['shuffle'], 
                **loader_kwargs
            )
        
        return self._active_loaders[key]

# class DataLoaderRegistry:
#     def __init__(self):
#         self.loaders = {}
    
#     def get_loader(self, data, mask, batch_size, mode):
#         # Create a unique key for this loader configuration
#         key = f"{id(data)}_{id(mask)}_{batch_size}_{mode}"
        
#         # Return cached loader if it exists
#         if key in self.loaders:
#             return self.loaders[key]
        
#         # Create appropriate loader parameters
#         if mode == 'train':
#             num_neighbors = [25, 10]
#             shuffle = True
#         elif mode == 'eval':
#             num_neighbors = [-1]  # Use all neighbors
#             shuffle = False
        

#         # Create loader
#         kwargs = {'batch_size': batch_size, 'num_workers': 6, 'persistent_workers': True}
#         loader = NeighborLoader(data, input_nodes=mask, num_neighbors=num_neighbors, shuffle=shuffle, **kwargs)
        
#         # Cache and return
#         self.loaders[key] = loader
#         return loader

# def batched_forward(model, data, batch_size, dropout, mode, mask, loader_registry):
#     original_training = model.training

#     log_logits = torch.zeros(data.x.size(0), model.outDim if model.skip_connections else model.outDim, dtype=data.x.dtype, device=data.x.device)

#     if loader_registry is None:
#         loader_registry = DataLoaderRegistry()

#     loader = loader_registry.get_loader(data, mask, batch_size, mode)


#     # if mode=='train':
#     #     # train loader
#     #     # loader = loader_registroy.get_loader(data, input_nodes=mask, num_neighbors=[25, 10], shuffle=True, **kwargs)
#     #     loader = loader_registry.get_loader(data, mask, batch_size, mode)
#     #     # loader = NeighborLoader(data, input_nodes=mask, num_neighbors=[25, 10], shuffle=True, **kwargs)
#     #     # batches = list(loader)

#     # elif mode=='eval':
#     #     # subgraph loader
#     #     # loader = NeighborLoader(copy.copy(data), input_nodes=mask, num_neighbors=[-1], shuffle=False, **kwargs)
#     #     loader = loader_registry.get_loader(data, mask, batch_size, mode)
#     #     # batches = list(loader)

#     # Perform forward pass on subgraph
#     with torch.set_grad_enabled(original_training):

#         for count, batch in enumerate(loader):
#             print(f'batch {count}/{len(loader)}')
#             outputs_all_layers = []
#             # for below -- [batch.n_id.to(x_all.device)].to(device) is technically better
#             batch_output = data.x[batch.n_id] 
#             print('batch_size:',batch_output.size())

#             for l in range(model.numLayers):
#                 batch_output = model.convs[l](batch_output, batch.edge_index)
#                 batch_output = model.activation_fn(batch_output)
#                 batch_output = F.dropout(batch_output, p=dropout, training=original_training)
#                 outputs_all_layers.append(batch_output)

#             # Handle skip connections
#             if model.skip_connections:
#                 output = torch.cat(outputs_all_layers, dim=-1)
#                 output = model.lin(output)
#             else:
#                 output = batch_output

#             # Apply log softmax
#             output = F.log_softmax(output, dim=1)
#             if mode=='train':
#                 loss = F.nll_loss(output, batch.y)
#                 loss.backward()
#                 optimizer.step()

#             if l==model.numLayers-1:
#                 log_logits[batch.n_id] = output 
            
#             del batch
#             del output
#             del batch_output

#     # Restore original training state
#     model.train(original_training)
#     return log_logits



def batched_forward_yesterday(model, x, edge_index, batch_size, dropout):
    # Prepare output tensor
    log_logits = torch.zeros(x.size(0), out_features if model.skip_connections else model.outDim, 
                            dtype=x.dtype, device=x.device)

    # Track which nodes have been processed
    processed_nodes = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
    
    # Get all unprocessed nodes
    all_nodes = torch.arange(x.size(0), device=x.device)
    print('# all nodes:',len(x))
    
    # Process batches until all nodes are covered
    while not processed_nodes.all():
        print('# unprocessed nodes:', (~processed_nodes).sum().item())
        # Get next batch of unprocessed nodes
        unprocessed = all_nodes[~processed_nodes]
        if len(unprocessed) == 0:
            break
            
        # Take a batch of unprocessed nodes
        batch_size_current = min(batch_size, len(unprocessed))
        batch_nodes = unprocessed[:batch_size_current]
        
        # Sample k-hop neighborhood for batch nodes
        subset, sub_edge_index, mapping, _ = k_hop_subgraph(
            batch_nodes, 
            1,  # Number of hops
            edge_index, 
            relabel_nodes=True
        )
        
        # Mark all nodes in this subset as processed
        processed_nodes[subset] = True
        
        # Create mapping from original indices to subgraph indices
        reverse_mapping = {int(original_idx.item()): sub_idx for sub_idx, original_idx in enumerate(subset)}
        
        # Map batch_nodes to their positions in the subgraph
        sub_batch_indices = torch.tensor([reverse_mapping[int(n.item())] for n in batch_nodes], 
                                         device=batch_nodes.device)
        
        # Prepare subgraph data
        sub_x = x[subset]
        
        # Original model's training state
        original_training = model.training
        
        # Prepare to collect intermediate outputs for skip connections
        intermediate_outputs = []
        
        # Perform forward pass on subgraph
        with torch.set_grad_enabled(original_training):
            # Replicate the original forward method logic
            sub_output = sub_x
            for l in range(model.numLayers):
                sub_output = model.convs[l](sub_output, sub_edge_index)
                sub_output = model.activation_fn(sub_output)
                sub_output = F.dropout(sub_output, p=dropout, training=original_training)
                intermediate_outputs.append(sub_output)
            
            # Handle skip connections
            if model.skip_connections:
                sub_output = torch.cat(intermediate_outputs, dim=-1)
                sub_output = model.lin(sub_output)
            
            # Apply log softmax
            sub_output = F.log_softmax(sub_output, dim=1)
        
        # Map output back to original node indices using the correct mapping
        log_logits[batch_nodes] = sub_output[sub_batch_indices]
        
        # For all other nodes in the subset, store their outputs too
        # This lets us avoid recomputing them in future batches
        for i, orig_idx in enumerate(subset):
            # Skip batch nodes as we've already handled them
            if orig_idx in batch_nodes:
                continue
            log_logits[orig_idx] = sub_output[i]

    # Restore original training state
    model.train(original_training)
    return log_logits


# def batched_forward_(model, x, edge_index, batch_size, dropout):
#     # Prepare output tensor
#     print('A')
#     log_logits = torch.zeros(x.size(0), out_features if model.skip_connections else model.outDim, 
#                                 dtype=x.dtype, device=x.device)

#     # Sample nodes for batching
#     all_nodes = torch.arange(x.size(0))
#     all_nodes = torch.randperm(x.size(0))  # Randomly permute nodes
#     batched_nodes = [
#         all_nodes[i:i + batch_size] 
#         for i in range(0, len(all_nodes), batch_size)
#     ]
#     print('batch 1:', batched_nodes[0])
#     print('len batched_nodes:',len(batched_nodes))

#     # Original model's training state
#     original_training = model.training

#     # Process each batch
#     for count, batch_nodes in enumerate(batched_nodes[:30]):
#         # Sample k-hop neighborhood for batch nodes
#         print(f'batch {count+1}/{len(batched_nodes)}')
#         subset, sub_edge_index, mapping, _ = k_hop_subgraph(
#             batch_nodes, 
#             2,  # Number of hops
#             edge_index, 
#             relabel_nodes=True
#         )
#         print('F')
        
#         # Prepare subgraph data
#         sub_x = x[subset]
#         print('sub_x size:',sub_x.size())
        
#         # Temporarily modify model's training state
#         model.train(original_training)
        
#         # Prepare to collect intermediate outputs for skip connections
#         intermediate_outputs = []
        
#         # Perform forward pass on subgraph
#         print("G")
#         with torch.set_grad_enabled(original_training):
#             # Replicate the original forward method logic
#             sub_output = sub_x
#             for l in range(model.numLayers):
#                 print("I")
#                 sub_output = model.convs[l](sub_output, sub_edge_index)
#                 sub_output = model.activation_fn(sub_output)
#                 sub_output = F.dropout(sub_output, p=dropout, training=original_training)
#                 intermediate_outputs.append(sub_output)
#                 print('J')
            
#             # Handle skip connections
#             if model.skip_connections:
#                 sub_output = torch.cat(intermediate_outputs, dim=-1)
#                 sub_output = model.lin(sub_output)
            
#             # Apply log softmax
#             sub_output = F.log_softmax(sub_output, dim=1)
        
#         # Map output back to original node indices
#         print('log_logits shape:',log_logits.size())
#         print('batch_nodes shape:',batch_nodes.size())
#         print('sub_output shape:',sub_output.size())
#         print('mapping shape:',mapping.size())
#         # log_logits[batch_nodes] = sub_output[mapping[batch_nodes]]
#         log_logits[batch_nodes] = sub_output[torch.arange(len(batch_nodes))]


#     # Restore original training state
#     model.train(original_training)
#     return log_logits




####

# class GraphBatchedForward:
#     @staticmethod
#     def batched_forward(
#         model, 
#         x, 
#         edge_index, 
#         dropout=0, 
#         batch_size=64, 
#         num_hops=2
#     ):
#         """
#         Batched forward pass that mimics the original forward method
        
#         Args:
#             model (torch.nn.Module): The original graph neural network model
#             x (torch.Tensor): Full node feature matrix
#             edge_index (torch.Tensor): Full graph edge indices
#             dropout (float, optional): Dropout rate
#             batch_size (int, optional): Size of each batch
#             num_hops (int, optional): Number of hops for neighborhood sampling
        
#         Returns:
#             torch.Tensor: Logits for all nodes
#         """
#         # Prepare output tensor
#         log_logits = torch.zeros(x.size(0), model.lin.out_features if model.skip_connections else model.convs[-1].out_channels, 
#                                  dtype=x.dtype, device=x.device)
        
#         # Sample nodes for batching
#         all_nodes = torch.arange(x.size(0))
#         batched_nodes = [
#             all_nodes[i:i + batch_size] 
#             for i in range(0, len(all_nodes), batch_size)
#         ]
#         # Original model's training state
#         original_training = model.training
        
#         # Process each batch
#         for batch_nodes in batched_nodes:
#             # Sample k-hop neighborhood for batch nodes
#             subset, sub_edge_index, mapping = k_hop_subgraph(
#                 batch_nodes, 
#                 num_hops, 
#                 edge_index, 
#                 relabel_nodes=True
#             )
            
#             # Prepare subgraph data
#             sub_x = x[subset]
            
#             # Temporarily modify model's training state
#             model.train(original_training)
            
#             # Prepare to collect intermediate outputs for skip connections
#             intermediate_outputs = []
            
#             # Perform forward pass on subgraph
#             with torch.set_grad_enabled(original_training):
#                 # Replicate the original forward method logic
#                 sub_output = sub_x
#                 for l in range(model.numLayers):
#                     sub_output = model.convs[l](sub_output, sub_edge_index)
#                     sub_output = model.activation_fn(sub_output)
#                     sub_output = F.dropout(sub_output, p=dropout, training=original_training)
#                     intermediate_outputs.append(sub_output)
                
#                 # Handle skip connections
#                 if model.skip_connections:
#                     sub_output = torch.cat(intermediate_outputs, dim=-1)
#                     sub_output = model.lin(sub_output)
                
#                 # Apply log softmax
#                 sub_output = F.log_softmax(sub_output, dim=1)
            
#             # Map output back to original node indices
#             log_logits[batch_nodes] = sub_output[mapping[batch_nodes]]
        
#         # Restore original training state
#         model.train(original_training)
        
#         return log_logits

# # Wrapper to integrate batching option
# def add_batching_method(model_class):
#     """
#     Decorator to add batched forward method to an existing model class
    
#     Args:
#         model_class (type): Original model class
    
#     Returns:
#         type: Modified model class with batched forward method
#     """
#     def batched_forward(
#         self, 
#         x, 
#         edge_index, 
#         dropout=0, 
#         use_batching=False, 
#         batch_size=64, 
#         num_hops=2
#     ):
#         """
#         Forward pass with optional batching
        
#         Args:
#             x (torch.Tensor): Node features
#             edge_index (torch.Tensor): Graph edge indices
#             dropout (float, optional): Dropout rate
#             use_batching (bool, optional): Enable batched processing
#             batch_size (int, optional): Size of each batch
#             num_hops (int, optional): Number of hops for neighborhood sampling
        
#         Returns:
#             torch.Tensor: Logits for all nodes
#         """
#         if use_batching:
#             return GraphBatchedForward.batched_forward(
#                 self, 
#                 x, 
#                 edge_index, 
#                 dropout=dropout, 
#                 batch_size=batch_size, 
#                 num_hops=num_hops
#             )
#         else:
#             # Original forward method
#             intermediate_outputs = []
#             for l in range(self.numLayers):
#                 x = self.convs[l](x, edge_index)
#                 x = self.activation_fn(x)
#                 x = F.dropout(x, p=dropout, training=self.training)
#                 intermediate_outputs.append(x)
#             if self.skip_connections:
#                 x = torch.cat(intermediate_outputs, dim=-1)
#                 x = self.lin(x)
#             return F.log_softmax(x, dim=1)
    
#     # Replace the original forward method
#     model_class.forward = batched_forward
#     return model_class

# Usage Example



####

''' consider this later for Flickr if still struggling '''
# import argparse
# import os.path as osp

# import torch_sparse
# import torch.nn.functional as F

# from torch_geometric.datasets import Flickr
# from torch_geometric.loader import GraphSAINTRandomWalkSampler
# from torch_geometric.nn import GraphConv
# from torch_geometric.typing import WITH_TORCH_SPARSE
# from torch_geometric.utils import degree

# if not WITH_TORCH_SPARSE:
#     quit("This example requires 'torch-sparse'")

# # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Flickr')
# # dataset = Flickr(path)
# data = dataset[0]
# row, col = data.edge_index
# data.edge_weight = 1. / degree(col, data.num_nodes)[col]  # Norm by in-degree.

# # parser = argparse.ArgumentParser()
# # parser.add_argument('--use_normalization', action='store_true')
# # args = parser.parse_args()

# loader = GraphSAINTRandomWalkSampler(data, batch_size=6000, walk_length=2,
#                                      num_steps=5, sample_coverage=100,
#                                      save_dir=dataset.processed_dir,
#                                      num_workers=4)


# class Net(torch.nn.Module):
#     def __init__(self, hidden_channels):
#         super().__init__()
#         in_channels = dataset.num_node_features
#         out_channels = dataset.num_classes
#         self.conv1 = GraphConv(in_channels, hidden_channels)
#         self.conv2 = GraphConv(hidden_channels, hidden_channels)
#         self.conv3 = GraphConv(hidden_channels, hidden_channels)
#         self.lin = torch.nn.Linear(3 * hidden_channels, out_channels)

#     def set_aggr(self, aggr):
#         self.conv1.aggr = aggr
#         self.conv2.aggr = aggr
#         self.conv3.aggr = aggr

#     def forward(self, x0, edge_index, edge_weight=None):
#         x1 = F.relu(self.conv1(x0, edge_index, edge_weight))
#         x1 = F.dropout(x1, p=0.2, training=self.training)
#         x2 = F.relu(self.conv2(x1, edge_index, edge_weight))
#         x2 = F.dropout(x2, p=0.2, training=self.training)
#         x3 = F.relu(self.conv3(x2, edge_index, edge_weight))
#         x3 = F.dropout(x3, p=0.2, training=self.training)
#         x = torch.cat([x1, x2, x3], dim=-1)
#         x = self.lin(x)
#         return x.log_softmax(dim=-1)


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Net(hidden_channels=256).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# use_normalization = True
# def train():
#     model.train()
#     model.set_aggr('add' if use_normalization else 'mean')

#     total_loss = total_examples = 0
#     for data in loader:
#         data = data.to(device)
#         optimizer.zero_grad()

#         if use_normalization:
#             edge_weight = data.edge_norm * data.edge_weight
#             out = model(data.x, data.edge_index, edge_weight)
#             loss = F.nll_loss(out, data.y, reduction='none')
#             loss = (loss * data.node_norm)[data.train_mask].sum()
#         else:
#             out = model(data.x, data.edge_index)
#             loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * data.num_nodes
#         total_examples += data.num_nodes
#     return total_loss / total_examples


# @torch.no_grad()
# def test():
#     model.eval()
#     model.set_aggr('mean')

#     out = model(data.x.to(device), data.edge_index.to(device))
#     pred = out.argmax(dim=-1)
#     correct = pred.eq(data.y.to(device))

#     accs = []
#     for _, mask in data('train_mask', 'val_mask', 'test_mask'):
#         accs.append(correct[mask].sum().item() / mask.sum().item())
#     return accs


# for epoch in range(1, 51):
#     loss = train()
#     accs = test()
#     print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {accs[0]:.4f}, '
#           f'Val: {accs[1]:.4f}, Test: {accs[2]:.4f}')


# import torch



# import torch
# from torch.optim import Optimizer
# import torch.nn.functional as F


# import torch
# from torch.optim import Optimizer

# class SAM(Optimizer):
#     def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
#         assert rho >= 0.0, f"Invalid rho value: {rho}"
#         defaults = dict(rho=rho, **kwargs)
#         params = list(params)  # Convert the generator to a list

#         super(SAM, self).__init__(params, defaults)
#         self.base_optimizer = base_optimizer(params, **kwargs)
#         self.param_groups = self.base_optimizer.param_groups

#         # Set rho for each parameter group
#         for group in self.param_groups:
#             group.setdefault('rho', rho)
        
#     @torch.no_grad()
#     def first_step(self, zero_grad=False):
#         grad_norm = self._grad_norm()
#         for group in self.param_groups:
#             scale = group['rho'] / (grad_norm + 1e-12)

#             for p in group['params']:
#                 if p.grad is None: continue
#                 e_w = p.grad * scale.to(p)
#                 p.add_(e_w)  # climb to the local maximum "w + e(w)"

#         if zero_grad: self.zero_grad()

#     @torch.no_grad()
#     def second_step(self, zero_grad=False):
#         for group in self.param_groups:
#             for p in group['params']:
#                 if p.grad is None: continue
#                 p.sub_(2 * p.grad * group['rho'])  # go back to the original point "w - e(w)"

#         self.base_optimizer.step()  # do the actual "sharpness-aware" update

#         if zero_grad: self.zero_grad()

#     def step(self, closure=None):
#         assert closure is not None, "SAM requires closure, but it was not provided"

#         closure = torch.enable_grad()(closure)
#         closure()
#         self.first_step(zero_grad=True)
#         closure()
#         self.second_step()
#         self.zero_grad()

#     def _grad_norm(self):
#         shared_device = self.param_groups[0]['params'][0].device
#         norm = torch.norm(
#             torch.stack([
#                 p.grad.norm(p=2).to(shared_device)
#                 for group in self.param_groups
#                 for p in group['params']
#                 if p.grad is not None
#             ]),
#             p=2
#         )
#         return norm
    


        # inDim = model_kwargs['inDim'] 
        # hDim = model_kwargs['hDim']
        # outDim = model_kwargs['outDim']
        # self.dropout = model_kwargs['dropout']
        # self.activation_fn = {'elu': F.elu, 'relu': F.relu}[model_kwargs['activation']]
        # self.numLayers = model_kwargs['numLayers'] 


        # conv_fn = {'GAT': GATConv, 'GCN': GCNConv, 'GraphConv': GraphConv, 'SAGE': SAGEConv, 'SGC': SGConv}[model_kwargs['arch']]
        # print('arch:',model_kwargs['arch'])
        # print('conv_fn:',conv_fn)
        # self.convs = nn.ModuleList()
        # if model_kwargs['arch']=='GAT':
        #     heads_1 = model_kwargs['heads_1']
        #     heads_2 = model_kwargs['heads_2']
        #     self.convs.append(conv_fn(in_channels=inDim, out_channels=hDim // heads_1, heads=heads_1, concat=True, dropout=self.dropout)) # First conv layer
        #     for l in range(self.numLayers - 2): # Intermediate conv layers
        #         self.convs.append(conv_fn(in_channels=hDim, out_channels=hDim // heads_2, heads=heads_2, concat=True, dropout=self.dropout))
        #     self.convs.append(conv_fn(in_channels=hDim, out_channels=outDim, heads=heads_2, concat=False, dropout=self.dropout)) # Final conv layer
            
        # elif model_kwargs['arch']!='GAT':



# class StudentSAGE(torch.nn.Module):
#     # for knowledge distillation -- so far just another GNN, maybe not anything special
#     def __init__(self, inDim, hDim, outDim, numLayers=2, dropout=0, activation_fn=F.elu, conv_fn=SAGEConv):
#         super(StudentSAGE, self).__init__()
#         self.numLayers = numLayers
#         self.dropout = dropout
#         self.activation_fn = activation_fn
#         self.convs = nn.ModuleList()

#         # First layer
#         self.convs.append(conv_fn(in_channels=inDim, out_channels=hDim))
#         # Intermediate layers (fewer compared to the teacher)
#         for l in range(numLayers - 1):  
#             self.convs.append(conv_fn(in_channels=hDim, out_channels=hDim))
#         # Final layer
#         self.convs.append(conv_fn(in_channels=hDim, out_channels=outDim))

#     def forward(self, x, edge_index):
#         for l in range(self.numLayers):
#             x = self.convs[l](x, edge_index)
#             x = self.activation_fn(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#         return F.log_softmax(x, dim=1)
    
# class NetWithKD(Net):
#     def __init__(self, teacher_model,**model_kwargs):
#         super(NetWithKD, self).__init__(**model_kwargs)
#         self.teacher_model = teacher_model  # Pre-trained teacher model
#         # self.temperature = model_kwargs.get('temperature', 5.0)  # Temperature for KD
#         # self.alpha = model_kwargs.get('alpha', 0.5)  # Weight for distillation loss

#     # def distillation_loss(self, student_logits, teacher_logits):
#     #     student_probs = F.log_softmax(student_logits / self.temperature, dim=1)
#     #     teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
#     #     return F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (self.temperature ** 2)

#     def forward(self, x, edge_index, y=None, train_mode=False):
#         student_logits = super().forward(x, edge_index)
#         if train_mode and y is not None:
#             with torch.no_grad():
#                 teacher_logits = self.teacher_model(x, edge_index)
#             kd_loss = self.distillation_loss(student_logits, teacher_logits)
#             ce_loss = F.cross_entropy(student_logits, y)
#             loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss
#             return student_logits, loss
#         return student_logits
    
    # def forward(self, x, edge_index, dropout=0):
    #     intermediate_outputs = []
    #     for l in range(self.numLayers):
    #         x = self.convs[l](x,edge_index)
    #         x = self.activation_fn(x)
    #         x = F.dropout(x, p=dropout, training=self.training)
    #         intermediate_outputs.append(x)
    #     if self.skip_connections == True:
    #         x = torch.cat(intermediate_outputs, dim=-1)
    #         x = self.lin(x)
    #     return F.log_softmax(x, dim=1)


### CAN PROB DELETE 


# class SAGE_special(torch.nn.Module):
#     def __init__(self, inDim, hDim, hDim_subgraphs, outDim, numLayers=2, numLayers_subgraphs=3, dropout=0, dropout_subgraphs=0, skip_connections=False, activation_fn=F.elu, conv_fn=SAGEConv):
#         super(SAGE_special, self).__init__()
#         self.numLayers = numLayers
#         self.dropout = dropout
#         self.dropout_subgraphs = dropout_subgraphs
#         self.skip_connections = skip_connections
#         self.activation_fn = activation_fn
        
#         self.convs = nn.ModuleList()
#         self.convs.append(conv_fn(in_channels=inDim, out_channels=hDim))
#         for l in range(numLayers - 2):
#             self.convs.append(conv_fn(in_channels=hDim, out_channels=hDim))
#         self.convs.append(conv_fn(in_channels=hDim, out_channels=outDim))

#         self.subgraph_convs = nn.ModuleList()
#         self.subgraph_convs.append(conv_fn(in_channels=inDim, out_channels=hDim_subgraphs))
#         for l in range(numLayers_subgraphs - 2):
#             self.subgraph_convs.append(conv_fn(in_channels=hDim_subgraphs, out_channels=hDim_subgraphs))
#         self.subgraph_convs.append(conv_fn(in_channels=hDim_subgraphs, out_channels=outDim))
#         if skip_connections:
#             self.lin = nn.Linear(hDim * numLayers, outDim)
    
#     def forward(self, x, edge_index, subgraphs=False):
#         convs = self.subgraph_convs if subgraphs else self.convs
#         dropout_rate = self.dropout_subgraphs if subgraphs else self.dropout  # Different dropout rates
#         intermediate_outputs = []
#         for l in range(self.numLayers):
#             x = convs[l](x, edge_index)
#             x = self.activation_fn(x)
#             x = F.dropout(x, p=dropout_rate, training=self.training)
#             intermediate_outputs.append(x)
#         if self.skip_connections:
#             x = torch.cat(intermediate_outputs, dim=-1)
#             x = self.lin(x)
#         return F.log_softmax(x, dim=1)


# class GraphConv_(torch.nn.Module):
#     def __init__(self, inDim, hDim, outDim, numLayers=2, dropout=0, skip_connections=False, activation_fn=F.relu):

#         super(GraphConv_, self).__init__()
#         self.numLayers=numLayers
#         self.dropout = dropout
#         self.skip_connections = skip_connections
#         self.activation_fn=activation_fn
#         self.convs = nn.ModuleList()

#         # First conv layer
#         self.convs.append(GraphConv(in_channels=inDim, out_channels=hDim))
#         # Intermediate conv layer
#         for l in range(numLayers - 2):
#             self.convs.append(GraphConv(in_channels=hDim,out_channels=hDim))
#         # Final conv layer
#         self.convs.append(GraphConv(in_channels=hDim, out_channels=outDim))
        
#         self.lin = torch.nn.Linear(self.numLayers * hDim, outDim)
        
#     def forward(self, x, edge_index):
#         intermediate_outputs = []
#         for l in range(self.numLayers):
#             x = self.convs[l](x,edge_index)
#             x = self.activation_fn(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#             intermediate_outputs.append(x)
#         if self.skip_connections == True:
#             x = torch.cat(intermediate_outputs, dim=-1)
#             x = self.lin(x)
#         return F.log_softmax(x, dim=1)

    


# class GCN_special(torch.nn.Module):
#     def __init__(self, inDim, hDim, hDim_subgraphs, outDim, numLayers=2, numLayers_subgraphs=3, dropout=0, dropout_subgraphs=0, skip_connections=False, activation_fn=F.relu):

#         super(GCN_special, self).__init__()
#         self.numLayers=numLayers
#         self.dropout = dropout
#         self.skip_connections=skip_connections
#         self.activation_fn=activation_fn
        
#         self.convs = nn.ModuleList()
#         conv1 = GCNConv(in_channels=inDim,out_channels=hDim)
#         self.convs.append(conv1)
#         for l in range(numLayers - 2):
#             self.convs.append(GCNConv(in_channels=hDim,out_channels=hDim))
#         self.convs.append(GCNConv(in_channels=hDim,out_channels=outDim))

#         self.convs_subgraphs = nn.ModuleList()
#         conv1 = GCNConv(in_channels=inDim,out_channels=hDim_subgraphs)
#         self.conv_subgraphs.append(conv1)
#         for l in range(numLayers_subgraphs- 2):
#             self.convs_subgraphs.append(GCNConv(in_channels=hDim_subgraphs,out_channels=hDim_subgraphs))
#         self.convs_subgraphs.append(GCNConv(in_channels=hDim_subgraphs,out_channels=outDim))
        
#     def forward(self, x, edge_index, subgraphs=False):
#         numLayers = self.numLayers if subgraphs==False else self.numLayers_subgraphs
#         dropout = self.dropout if subgraphs == False else self.dropout_subgraphs
#         convs = self.convs if subgraphs==False else self.convs_subgraphs
#         intermediate_outputs = []
#         for l in range(numLayers):
#             x = convs[l](x,edge_index)
#             x = self.activation_fn(x)
#             x = F.dropout(x, p=dropout, training=self.training)
#             intermediate_outputs.append(x)
#         if self.skip_connections == True:
#             x = torch.cat(intermediate_outputs, dim=-1)
#             x = self.lin(x)
#         return F.log_softmax(x, dim=1)


# class SAGE(torch.nn.Module):
#     def __init__(self, inDim, hDim, outDim, numLayers=2, dropout=0, skip_connections=False, activation_fn=F.elu, conv_fn=SAGEConv):

#         super(SAGE, self).__init__()
#         self.numLayers=numLayers
#         self.dropout = dropout
#         self.skip_connections=skip_connections
#         self.activation_fn=activation_fn
#         self.convs = nn.ModuleList()

#         # First layer
#         self.convs.append(SAGEConv(in_channels=inDim, out_channels=hDim))
#         # Intermediate layers
#         for l in range(numLayers - 2):
#             self.convs.append(SAGEConv(in_channels=hDim, out_channels=hDim))
#         # Final layer
#         self.convs.append(SAGEConv(in_channels=hDim, out_channels=outDim))
        
#     def forward(self, x, edge_index):
#         intermediate_outputs = []
#         for l in range(self.numLayers):
#             x = self.convs[l](x,edge_index)
#             x = self.activation_fn(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#             intermediate_outputs.append(x)
#         if self.skip_connections == True:
#             x = torch.cat(intermediate_outputs, dim=-1)
#             x = self.lin(x)
#         return F.log_softmax(x, dim=1)


# class GAT(nn.Module):
#     def __init__(self, inDim, hDim, outDim,
#                  heads_1=8, heads_2=1, attDrop=0, inDrop=0, numLayers=2, skip_connections=False, activation_fn=F.elu):
#         super(GAT, self).__init__()
#         self.attDrop = attDrop
#         self.inDrop = inDrop
#         self.numLayers=numLayers
#         self.skip_connections=skip_connections
#         self.activation_fn = activation_fn
#         self.convs = nn.ModuleList()

#         # First layer
#         conv1 = GATConv(in_channels=inDim,
#                         out_channels=hDim // heads_1,
#                         heads=heads_1,
#                         concat=True,
#                         dropout=attDrop)
#         self.convs.append(conv1)

#         # Intermediate layers
#         for l in range(numLayers - 2):
#             self.convs.append(GATConv(in_channels=hDim,
#                                       out_channels=hDim // heads_2,
#                                       heads=heads_2,
#                                       concat=True,
#                                       dropout=attDrop))

#         # Final layer
#         self.convs.append(GATConv(in_channels=hDim,
#                                   out_channels=outDim,
#                                   heads=heads_2,
#                                   concat=False,
#                                   dropout=attDrop))

#     def forward(self, x, edge_index):
#         intermediate_outputs = []
#         for l in range(self.numLayers):
#             x = self.convs[l](x,edge_index)
#             x = self.activation_fn(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#             intermediate_outputs.append(x)
#         if self.skip_connections == True:
#             x = torch.cat(intermediate_outputs, dim=-1)
#             x = self.lin(x)
#         return F.log_softmax(x, dim=1)



# class GCN(torch.nn.Module):
#     def __init__(self, inDim, hDim, outDim, numLayers=2, dropout=0,skip_connections=False, activation_fn=F.relu):

#         super(GCN, self).__init__()
#         self.numLayers=numLayers
#         self.dropout = dropout
#         self.skip_connections=skip_connections
#         self.activation_fn=activation_fn
#         self.convs = nn.ModuleList()

#         # First layer
#         conv1 = GCNConv(in_channels=inDim,
#                         out_channels=hDim)
#         self.convs.append(conv1)

#         # Intermediate layers
#         for l in range(numLayers - 2):
#             self.convs.append(GCNConv(in_channels=hDim,
#                                       out_channels=hDim))

#         # Final layer
#         self.convs.append(GCNConv(in_channels=hDim,
#                                   out_channels=outDim))
        
#     def forward(self, x, edge_index):
#         intermediate_outputs = []
#         for l in range(self.numLayers):
#             x = self.convs[l](x,edge_index)
#             x = self.activation_fn(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#             intermediate_outputs.append(x)
#         if self.skip_connections == True:
#             x = torch.cat(intermediate_outputs, dim=-1)
#             x = self.lin(x)
#         return F.log_softmax(x, dim=1)
    
