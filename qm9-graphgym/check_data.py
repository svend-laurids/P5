from collections import Counter
from torch_geometric.datasets import QM9

ds = QM9('data/QM9/QM9')
print(ds)
print('x:', ds[0].x.shape, ' edge_attr:', ds[0].edge_attr.shape, ' y:', ds[0].y.shape)

atom = ds._data.x[:, :5].argmax(1)
bond = ds._data.edge_attr.argmax(1)
print('atom types (H,C,N,O,F):', sorted(Counter(atom.tolist()).items()))
print('bond types (single,double,triple,aromatic):', sorted(Counter(bond.tolist()).items()))

combos = Counter()
for d in ds[:5000]:
    a, b = d.x[:, :5].argmax(1), d.edge_attr.argmax(1)
    for s, t, e in zip(d.edge_index[0], d.edge_index[1], b):
        combos[(a[t].item(), e.item(), a[s].item())] += 1
print(len(combos), 'of 100 type combinations occur in 5000 molecules')
print('rarest:', sorted(combos.items(), key=lambda kv: kv[1])[:10])