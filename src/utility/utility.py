import torch
from torch_geometric.utils import to_networkx
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import QM9

import matplotlib.pyplot as plt
import networkx as nx
import orbit_count

from pathlib import Path
path_root = Path(__file__).parent.parent.parent
path_QM9 = path_root.joinpath('data/QM9')
path_QM9_type_annotated = path_QM9.joinpath('type-annotated/data.pt')

def load_QM9():
    return QM9(root=path_QM9)

def load():
    dataset = InMemoryDataset()
    dataset.load(path_QM9_type_annotated)
    return dataset 

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
