import torch
from torch_geometric.datasets import QM7b
from torch_geometric.graphgym.register import register_loader
from torch_geometric.data import InMemoryDataset

from pathlib import Path
path_root = Path(__file__).parent.parent.parent.parent.parent.parent
path_QM9_annotated = path_root.joinpath('data/QM9/type-annotated/data.pt')

@register_loader('qm9-annotated')
def load_dataset_qm9_annotated(format, name, dataset_dir):

    dataset = InMemoryDataset()
    dataset.load(path_QM9_annotated)

    n = len(dataset)
    indices = torch.randperm(n)

    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    splits = {
        'train_graph_index': train_idx,
        'val_graph_index': val_idx,
        'test_graph_index': test_idx,
    }

    for key, value in splits.items():
        dataset._data[key] = value
        dataset.slices[key] = torch.tensor(
            [0, len(value)],
            dtype=torch.long
        )

    dataset._data_list = None

    return dataset

