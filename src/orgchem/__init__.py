import matplotlib.pyplot as plt
import networkx as nx
import orbit_count

import torch
import torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.nn import MessagePassing 
from torch.nn import ModuleList
from torch.nn import Linear
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
from torch_geometric.nn import RGCNConv
from torch_geometric.nn import GCNConv 

from pathlib import Path
path_root = Path(__file__).parent.parent.parent
path_QM9 = path_root.joinpath('data/QM9')
path_QM9_type_annotated = path_QM9.joinpath('type-annotated/data.pt')

#class TAggregation(Aggregation):
#    def forward(self, x, index, ptr=None, dim_size=None, dim=-2):
#        print(x)
#        print(index)
#        return self.reduce(
#            x,
#            index,
#            ptr=ptr,
#            dim_size=dim_size,
#            dim=dim,
#            reduce="sum",
#        )

class TypedTransform:
    def __init__(self):
        pass

    def type_to_transform(node_type1, edge_type, node_type2):
        pass

class UnitTransform(TypedTransform):
    def __init__(self, in_channels, out_channels):
        self.linear = Linear(in_channels, out_channels, bias=False)

    def type_to_transform(node_type1, edge_type, node_type2):
        return [self.linear for i in edge_type]

class TypedMessagePassing(MessagePassing):
    def __init__(self, in_channels, out_channels, type_transformer): #, n_types=1, typefunc=None, edge_types=None, node_types=None):
        super().__init__(aggr=None)
        self.type_transformer = type_transformer
        #self.typefunc = typefunc

    def forward(self, graph):
        return self.propagate(graph.edge_index, x=graph.x, edge_type=graph.edge_type, node_type=graph.node_type) #, typemap=typemap)

    def message(self, x_j):
        return x_j

    def aggregate(self, x, index, edge_index, ptr=None, dim_size=None, dim=-2, edge_type=None, node_type=None):
        transform = self.type_transformer.transform(node_type[edge_index[0]], edge_type, node_type[edge_index[1]])

        y = torch.bmm(
            x.unsqueeze(1),
            transform
        ).squeeze(1) 

        out = torch.zeros(dim_size, y.size(1))
        out.index_add_(0, index, y)
    
        return out

class UnitMessagePassing(TypedMessagePassing):
    def __init__(self, in_channels, out_channels, type_transformer): #, n_types=1, typefunc=None, edge_types=None, node_types=None):
        super().__init__(aggr=None)
        self.type_transformer = type_transformer
        #self.typefunc = typefunc

    def forward(self, graph):
        return self.propagate(graph.edge_index, x=graph.x, edge_type=graph.edge_type, node_type=graph.node_type) #, typemap=typemap)

    def message(self, x_j):
        return x_j

    def aggregate(self, x, index, edge_index, ptr=None, dim_size=None, dim=-2, edge_type=None, node_type=None):
        transform = self.type_transformer.transform(node_type[edge_index[0]], edge_type, node_type[edge_index[1]])

        y = torch.bmm(
            x.unsqueeze(1),
            transform
        ).squeeze(1) 

        out = torch.zeros(dim_size, y.size(1))
        out.index_add_(0, index, y)
    
        return out


def graph_plot(graph):
    G = to_networkx(graph)
    nx.draw(G, with_labels=True, node_color='skyblue', node_size=1000, font_size=14, edge_color='gray')
    plt.show()

def graph_graphlets(graph, n):
    G = nx.Graph(to_networkx(graph))
    node_count = orbit_count.node_orbit_counts(G, graphlet_size=n)
    return node_count

def graph_degrees(graph):
    degrees = torch.zeros(graph.num_nodes)
    for edge in graph.edge_index[0]:
        degrees[edge] += 1
    return degrees

def process(dataset, path):
    type_annotated_dataset = [] 

    for i, graph in enumerate(dataset):
        graph['node_type'] = graph.pop('z')

        graph.degrees = graph_degrees(graph)

        graph.graphlet4 = torch.tensor(graph_graphlets(graph, 4))
                    
        graph.graphlet5 = torch.tensor(graph_graphlets(graph, 5))

        graph.edge_type = torch.zeros(graph.edge_attr.shape[0])
        for n_edge, edge in enumerate(graph.edge_attr):
            for n_attr, attr in enumerate(edge):
                if attr > 0:
                    graph.edge_type[n_edge] = n_attr

        type_annotated_dataset.append(graph)
        if i % round(dataset.len()/100.0) == 0:
            print(f"{(i/dataset.len())*100.0}%")
            print(f"{graph}")

    InMemoryDataset.save(type_annotated_dataset, path)
    print("Done!")

def stats(dataset):
    stats = { 
        degree : 0,
        graphlet4 : 0,
        graphlet5 : 0 
    }

    for i, graph in enumerate(dataset):
        for degree in graph.degrees:
            stats['degree'] = stats['degree'] + 1

        for node in graph.graphlet4:
            for n, motif in enumerate(node):
                if motif > 0:
                    stats['graphlet4'] = stats['graphlet4'] + 1
                    
        for node in graph.graphlet5:
            for n, motif in enumerate(node):
                if motif > 0:
                    stats['graphlet5'] = stats['graphlet5'] + 1

    return stats


def load(path):
    dataset = InMemoryDataset()
    dataset.load(path)
    return dataset 

def main() -> None:
    #dataset = QM9(root=path_QM9)
    #process(dataset, path_QM9_type_annotated)
    
    dataset = load(path_QM9_type_annotated)
    print(dataset[0]['node_type'])
    #tgcn = TGCN(1, 1) #, 1)
    #tgcn(dataset[0])
      
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
