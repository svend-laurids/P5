from utility.utility import load, load_QM9

#import torch
#import torch.nn.functional as F
from torch_geometric.datasets import QM9
#from torch_geometric.nn import MessagePassing 
#from torch.nn import ModuleList
#from torch.nn import Linear
#from torch_geometric.nn.aggr import Aggregation
#from torch_geometric.data import InMemoryDataset
#from torch_geometric.loader import DataLoader
#from torch_geometric.utils import to_networkx
#from torch_geometric.nn import RGCNConv
#from torch_geometric.nn import GCNConv 

from .tgnn2 import *

def main() -> None:
    dataset = load()
    tgnn1 = LinearTGNN(11, 3)
    tgnn2 = AtomTGNN(11, 3)
    tgnn3 = BondTGNN(11, 3)
    tgnn4 = BondAtomSumTGNN(11, 3)
    print(tgnn1(dataset[3]))
    print(tgnn2(dataset[3]))
    print(tgnn3(dataset[3]))
    print(tgnn4(dataset[3]))
      
    #print(stats_degree)
    #print(dataset.stats_graphlet4)
    #print(dataset.stats_graphlet5)

    #x()
    #y()
    #model = GCN(dataset.num_features, dataset.num_classes)
    #loader = DataLoader(dataset, batch_size=1, shuffle=False)
    #for batch in loader:
    #    print(batch[0])
    #    out = model(batch[0])
    #    print(out)
    #    break

    #print(dataset[0])
    #print(dataset[0].z)
    #print(dataset[0].x)
        
    #print(graph_degrees(dataset[0]))

    #optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    #model.train()
    #for epoch in range(200):
    #    optimizer.zero_grad()
    #    out = model(data)
    #    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    #    loss.backward()
    #    optimizer.step()



