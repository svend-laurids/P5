import torch
from torch_geometric.datasets import QM9
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loader import set_dataset_attr
from torch_geometric.graphgym.register import register_loader

def add_types(data):
    data.node_type = data.x[:, :5].argmax(-1)     # atom type
    data.edge_type = data.edge_attr.argmax(-1)    # bond type
    return data

@register_loader('qm9_typed')
def load_qm9(format, name, dataset_dir):
    if format != 'PyG' or name != 'QM9':
        return None

    ds = QM9(f'{dataset_dir}/{name}', transform=add_types)

    # keep one target, standardised
    t = cfg.qm9.target
    y = ds._data.y[:, t:t + 1]
    ds._data.y = (y - y.mean()) / y.std()

    if cfg.qm9.subset > 0:                        # debugging only
        ds = ds[:cfg.qm9.subset]

    # fixed random split, independent of the run seed,
    # so every weight-sharing mode sees the same train/val/test molecules
    n = len(ds)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    n_train = int(cfg.dataset.split[0] * n)
    n_val = int(cfg.dataset.split[1] * n)
    splits = {
        'train_graph_index': perm[:n_train],
        'val_graph_index': perm[n_train:n_train + n_val],
        'test_graph_index': perm[n_train + n_val:],
    }
    for key, idx in splits.items():
        set_dataset_attr(ds, key, idx, len(idx))

    return ds