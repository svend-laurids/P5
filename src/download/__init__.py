#import torch
from torch_geometric.datasets import QM9

from pathlib import Path
path_root = Path(__file__).parent.parent.parent
path_QM9 = path_root.joinpath('data/QM9')

def download(path):
    return QM9(root=path_QM9)

def main():
    dataset = download(path_QM9)
