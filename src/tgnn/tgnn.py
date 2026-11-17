import torch
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing 
from torch_geometric.utils import scatter

from torch.nn import ModuleList, Linear

class Aggregator:
    def __init__(self, in_channels, out_channels, reduce='sum'):
        pass

    def aggregate(self, node_type1, edge_type, node_type2):
        pass

class TGNN(MessagePassing):
    def __init__(self, aggregator): #, n_types=1, typefunc=None, edge_types=None, node_types=None):
        super().__init__(aggr=None)
        self.aggregator = aggregator

    def forward(self, graph):
        print(graph)
        return self.propagate(graph.edge_index, 
                              x=graph.x, 
                              edge_type=graph.edge_type, 
                              node_type=graph.node_type,
                              node_degree=graph.degrees,
                              node_graphlet4=graph.graphlet4,
                              node_graphlet5=graph.graphlet5) 

    def message(self, x_j, x):
        print(x)
        print(x_j)
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

        return self.aggregator.aggregate(x,
                                         edge_index, 
                                         edge_type, 
                                         node_type, 
                                         node_degrees, 
                                         node_graphlet4, 
                                         node_graphlet5)

class LinearAggregator(Aggregator):
    def __init__(self, in_channels, out_channels, reduce='sum'):
        self.linear = Linear(in_channels, out_channels, bias=False)
        self.reduce = reduce

    def aggregate(self, 
                  x,
                  edge_index, 
                  edge_type, 
                  node_type, 
                  node_degrees, 
                  node_graphlet4, 
                  node_graphlet5):

        messages = self.linear(x)
        print(edge_type)

        return scatter(messages, edge_index[1], dim=0, dim_size=node_type.size(0), reduce=self.reduce)

class BondAggregator(Aggregator):
    def __init__(self, in_channels, out_channels, reduce='sum'):
        self.weights = torch.randn(4, in_channels, out_channels)
        self.reduce = reduce

    def aggregate(self, 
                  x,
                  edge_index, 
                  edge_type, 
                  node_type, 
                  node_degrees, 
                  node_graphlet4, 
                  node_graphlet5):

        messages = torch.bmm(x.unsqueeze(1), self.weights[edge_type]).squeeze(1) 

        return scatter(messages, edge_index[1], dim=0, dim_size=node_type.size(0), reduce=self.reduce)

class AtomAggregator(Aggregator):
    def __init__(self, in_channels, out_channels, reduce='sum'):
        self.weights = torch.randn(5, in_channels, out_channels)
        self.reduce = reduce

    def aggregate(self, 
                  x,
                  edge_index, 
                  edge_type, 
                  node_type, 
                  node_degrees, 
                  node_graphlet4, 
                  node_graphlet5):

        messages = torch.bmm(x.unsqueeze(1), self.weights_atom[node_type[edge_index[1]]]).squeeze(1) 

        return scatter(messages, edge_index[1], dim=0, dim_size=node_type.size(0), reduce=self.reduce)

class BondAtomSumAggregator(Aggregator):
    def __init__(self, in_channels, out_channels, reduce='sum'):
        self.weights_bond = torch.randn(4, in_channels, out_channels)
        self.weights_atom = torch.randn(5, in_channels, out_channels)
        self.reduce = reduce

    def aggregate(self, 
                  x,
                  edge_index, 
                  edge_type, 
                  node_type, 
                  node_degrees, 
                  node_graphlet4, 
                  node_graphlet5):

        messages_bond = torch.bmm(x.unsqueeze(1), self.weights_bond[edge_type]).squeeze(1) 
        messages_atom = torch.bmm(x.unsqueeze(1), self.weights_atom[node_type[edge_index[1]]]).squeeze(1) 

        messages_sum = messages_bond + messages_atom

        return scatter(messages_sum, edge_index[1], dim=0, dim_size=node_type.size(0), reduce=self.reduce)
