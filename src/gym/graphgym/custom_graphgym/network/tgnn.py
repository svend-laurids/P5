import torch
import torch.nn.functional as F

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.init import init_weights
from torch_geometric.graphgym.models.layer import (
    BatchNorm1dNode,
    GeneralLayer,
    GeneralMultiLayer,
    new_layer_config,
)
from torch_geometric.graphgym.register import register_stage

from msg import LinearMessenger, BondMessenger, AtomMessenger, BondAtomSumMessenger, 

class TGNNMessagePassing(MessagePassing) -> GeneralLayer:
    def __init__(self, dim_in: int, dim_out: int, has_act: bool = True):
        super().__init__()

    def forward(self, graph):
        return self.propagate(graph.edge_index, 
                              x=graph.x, 
                              edge_type=graph.edge_type, 
                              node_type=graph.node_type,
                              node_degree=graph.degrees,
                              node_graphlet4=graph.graphlet4,
                              node_graphlet5=graph.graphlet5) 

    def message(self, x_j, x):
        return x_j

    def aggregate(self, 
                  x, 
                  index, 
                  edge_index, 
                  ptr=None, 
                  dim_size=None, 
                  dim=-2, 
                  edge_type=None, 
                  node_type=None,
                  node_degrees=None,
                  node_graphlet4=None,
                  node_graphlet5=None):

        return self.messenger.message(x,
                                      edge_index, 
                                      edge_type, 
                                      node_type, 
                                      node_degrees, 
                                      node_graphlet4, 
                                      node_graphlet5)
    
def TGNNLayer(dim_in: int, dim_out: int, has_act: bool = True) -> GeneralLayer:
    return GeneralLayer(
        cfg.gnn.layer_type,
        layer_config=new_layer_config(
            dim_in,
            dim_out,
            1,
            has_act=has_act,
            has_bias=False,
            cfg=cfg,
        ),
    )

@register_head('readout')
class Readout(torch.nn.Module):


@register_stage('stack')
@register_stage('skipsum')
@register_stage('skipconcat')
class TMPNN(torch.nn.Module):
    def __init__(self, dim_in, dim_out, num_layers):
        super().__init__()
        self.num_layers = num_layers
        for i in range(num_layers):
            if cfg.tgnn.stage_type == 'skipconcat':
                d_in = dim_in if i == 0 else dim_in + i * dim_out
            else:
                d_in = dim_in if i == 0 else dim_out
            match cfg.tgnn.layer_type:
                case 'linear':
                    self.add_module(f'layer{i}', LinearTGNN(dim_in, dim_out))
                case 'bond':
                    self.add_module(f'layer{i}', BondTGNN(dim_in, dim_out))
                case 'atom':
                    self.add_module(f'layer{i}', AtomTGNN(dim_in, dim_out))
                case 'atombondsum':
                    self.add_module(f'layer{i}', AtomBondSumTGNN(dim_in, dim_out))
                case _:
                    print(f"Error: cfg.tgnn.msg = {cfg.tgnn.msg}, which is not a valid messenger")


    def forward(self, batch):
        for i, layer in enumerate(self.children()):
            x = batch.x
            batch = layer(batch)
            if cfg.tgnn.stage_type == 'skipsum':
                batch.x = x + batch.x
            elif (cfg.tgnn.stage_type == 'skipconcat' and i < self.num_layers - 1):
                batch.x = torch.cat([x, batch.x], dim=1)
        if cfg.tgnn.l2norm:
            batch.x = F.normalize(batch.x, p=2, dim=-1)
        return batch

class TGNN(torch.nn.Module):
    def __init__(self, dim_in: int, dim_out: int, **kwargs):
        super().__init__()

        TMPNN = register.stage_dict[cfg.tgnn.stage_type]
        Readout = register.head_dict[cfg.tgnn.head]

        self.mpnn = TMPNN(dim_in=dim_in, dim_out=cfg.tgnn.dim_inner, num_layers=cfg.tgnn.layers_mp)
        self.readout = Readout(dim_in=cfg.tgnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children(): 
            batch = module(batch)

        return batch
