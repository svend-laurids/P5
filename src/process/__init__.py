import networkx as nx
import orbit_count

import torch
from torch_geometric.datasets import QM9
from torch_geometric.utils import to_networkx
from torch_geometric.data import InMemoryDataset

from pathlib import Path
path_root = Path(__file__).parent.parent.parent
path_QM9 = path_root.joinpath('data/QM9')
path_QM9_type_annotated = path_QM9.joinpath('type-annotated/data.pt')

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
        graph.degrees = graph_degrees(graph)

        graph.graphlet4 = torch.tensor(graph_graphlets(graph, 4))
                    
        graph.graphlet5 = torch.tensor(graph_graphlets(graph, 5))

        graph.edge_type = torch.zeros(graph.edge_attr.shape[0], dtype=torch.int)
        for n_edge, edge in enumerate(graph.edge_attr):
            for n_attr, attr in enumerate(edge):
                if attr > 0:
                    graph.edge_type[n_edge] = n_attr

        graph.node_type = torch.zeros_like(graph.z)
        for n_z, z in enumerate(graph.z):
            match z:
                case 0:
                    graph.node_type[n_z] = 0
                case 6:
                    graph.node_type[n_z] = 1
                case 7:
                    graph.node_type[n_z] = 2
                case 8:
                    graph.node_type[n_z] = 3
                case 9:
                    graph.node_type[n_z] = 4 

        type_annotated_dataset.append(graph)
        if i % round(dataset.len()/100.0) == 0:
            print(f"{(i/dataset.len())*100.0}%")
            print(f"{graph}")

    InMemoryDataset.save(type_annotated_dataset, path)
    print("Done!")

def main():
    dataset = QM9(root=path_QM9)
    process(dataset, path_QM9_type_annotated)
