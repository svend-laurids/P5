import torch
import torch.nn as nn
from torch.nn import Parameter

from torch_geometric.nn import MessagePassing 
from torch_geometric.utils import scatter

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_layer

@register_layer('tgnn_linear')
class LinearTGNN(MessagePassing):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear = torch.rand(dim_in, dim_out)

    def forward(self, graph):
        return self.propagate(graph.edge_index, 
                              x=graph.x, 
                              edge_type=graph.edge_type, 
                              node_type=graph.node_type,
                              node_degree=graph.degrees,
                              node_graphlet4=graph.graphlet4,
                              node_graphlet5=graph.graphlet5) 

    def message(self, x_j, x, edge_type=None, node_type=None, node_degree=None, node_graphlet4=None, node_graphlet5=None):
        messages = x_j @ self.linear
        return messages

    def aggregate(self, x, index, edge_index, ptr=None, dim_size=None, dim=-2, node_type=None):
        return scatter(x, edge_index[1], dim=0, dim_size=node_type.size(0), reduce=cfg.tgnn_config.aggregation)

@register_layer('tgnn_bond')
class BondTGNN(MessagePassing):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.weights = torch.randn(5, dim_in, dim_out)

    def forward(self, graph):
        return self.propagate(graph.edge_index, 
                              x=graph.x, 
                              edge_type=graph.edge_type, 
                              node_type=graph.node_type,
                              node_degree=graph.degrees,
                              node_graphlet4=graph.graphlet4,
                              node_graphlet5=graph.graphlet5) 

    def message(self, x_j, x, edge_type=None, node_type=None, node_degree=None, node_graphlet4=None, node_graphlet5=None):
        messages = torch.bmm(x_j.unsqueeze(1), self.weights[edge_type]).squeeze(1) 
        return messages

    def aggregate(self, x, index, edge_index, ptr=None, dim_size=None, dim=-2, node_type=None):
        return scatter(x, edge_index[1], dim=0, dim_size=node_type.size(0), reduce=cfg.tgnn_config.aggregation)

@register_layer('tgnn_atom')
class AtomTGNN(MessagePassing):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.weights = torch.randn(4, dim_in, dim_out)

    def forward(self, graph):
        return self.propagate(graph.edge_index, 
                              x=graph.x, 
                              edge_type=graph.edge_type, 
                              node_type=graph.node_type,
                              node_degree=graph.degrees,
                              node_graphlet4=graph.graphlet4,
                              node_graphlet5=graph.graphlet5) 

    def message(self, x_j, x, edge_index, edge_type=None, node_type=None, node_degree=None, node_graphlet4=None, node_graphlet5=None):
        messages = torch.bmm(x_j.unsqueeze(1), self.weights[node_type[edge_index[1]]]).squeeze(1) 
        return messages

    def aggregate(self, x, index, edge_index, ptr=None, dim_size=None, dim=-2, node_type=None):
        return scatter(x, edge_index[1], dim=0, dim_size=node_type.size(0), reduce=cfg.tgnn_config.aggregation)

@register_layer('tgnn_atombondsum')
class AtomBondSumTGNN(MessagePassing):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.weights_bond = torch.randn(4, dim_in, dim_out)
        self.weights_atom = torch.randn(5, dim_in, dim_out)

    def forward(self, graph):
        return self.propagate(graph.edge_index, 
                              x=graph.x, 
                              edge_type=graph.edge_type, 
                              node_type=graph.node_type,
                              node_degree=graph.degrees,
                              node_graphlet4=graph.graphlet4,
                              node_graphlet5=graph.graphlet5) 

    def message(self, x_j, x, edge_index, edge_type=None, node_type=None, node_degree=None, node_graphet4=None, node_graphlet5=None):
        messages_bond = torch.bmm(x_j.unsqueeze(1), self.weights_bond[edge_type]).squeeze(1) 
        messages_atom = torch.bmm(x_j.unsqueeze(1), self.weights_atom[node_type[edge_index[1]]]).squeeze(1) 
        return messages_bond + messages_atom

    def aggregate(self, x, index, edge_index, ptr=None, dim_size=None, dim=-2, node_type=None):
        return scatter(x, edge_index[1], dim=0, dim_size=node_type.size(0), reduce=cfg.tgnn_config.aggregation)
