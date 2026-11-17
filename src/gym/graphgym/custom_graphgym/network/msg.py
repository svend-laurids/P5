import torch
import torch.nn.functional as F

from torch_geometric.nn import MessengerPassing 
from torch_geometric.utils import scatter

from torch.nn import ModuleList, Linear

class Messenger:
    def __init__(self, in_channels, out_channels, reduce='sum'):
        pass

    def message(self, node_type1, edge_type, node_type2):
        pass

class LinearMessenger(Messenger):
    def __init__(self, in_channels, out_channels, reduce='sum'):
        self.linear = Linear(in_channels, out_channels, bias=False)
        self.reduce = reduce

    def message(self, 
                  x,
                  edge_index, 
                  edge_type, 
                  node_type, 
                  node_degrees, 
                  node_graphlet4, 
                  node_graphlet5):

        messages = self.linear(x)

        return scatter(messages, edge_index[1], dim=0, dim_size=node_type.size(0), reduce=self.reduce)

class BondMessenger(Messenger):
    def __init__(self, in_channels, out_channels, reduce='sum'):
        self.weights = torch.randn(4, in_channels, out_channels)
        self.reduce = reduce

    def message(self, 
                  x,
                  edge_index, 
                  edge_type, 
                  node_type, 
                  node_degrees, 
                  node_graphlet4, 
                  node_graphlet5):

        messages = torch.bmm(x.unsqueeze(1), self.weights[edge_type]).squeeze(1) 

        return scatter(messages, edge_index[1], dim=0, dim_size=node_type.size(0), reduce=self.reduce)

class AtomMessenger(Messenger):
    def __init__(self, in_channels, out_channels, reduce='sum'):
        self.weights = torch.randn(5, in_channels, out_channels)
        self.reduce = reduce

    def message(self, 
                  x,
                  edge_index, 
                  edge_type, 
                  node_type, 
                  node_degrees, 
                  node_graphlet4, 
                  node_graphlet5):

        messages = torch.bmm(x.unsqueeze(1), self.weights_atom[node_type[edge_index[1]]]).squeeze(1) 

        return scatter(messages, edge_index[1], dim=0, dim_size=node_type.size(0), reduce=self.reduce)

class BondAtomSumMessenger(Messenger):
    def __init__(self, in_channels, out_channels, reduce='sum'):
        self.weights_bond = torch.randn(4, in_channels, out_channels)
        self.weights_atom = torch.randn(5, in_channels, out_channels)
        self.reduce = reduce

    def message(self, 
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
