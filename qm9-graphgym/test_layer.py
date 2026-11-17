import torch
from torch_geometric.graphgym.config import cfg, set_cfg
set_cfg(cfg)
cfg.gnn.agg = 'add'
from custom.typed_layer import TypedConv

torch.manual_seed(0)
N, d_in, d_out, nt, ne = 6, 3, 2, 5, 4
x = torch.randn(N, d_in)
node_type = torch.tensor([0, 1, 1, 2, 3, 4])
src = torch.tensor([0, 1, 2, 3, 4, 5, 1, 3])          # sender u
dst = torch.tensor([1, 0, 3, 2, 5, 4, 2, 1])          # receiver v
edge_type = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
edge_index = torch.stack([src, dst])

for mode in ['shared', 'edge', 'node', 'sum', 'product']:
    conv = TypedConv(d_in, d_out, mode)
    out = conv(x, edge_index, node_type, edge_type)

    manual = conv.W0(x).clone()                        # self term
    for u, v, e in zip(src.tolist(), dst.tolist(), edge_type.tolist()):
        tv, tu = node_type[v].item(), node_type[u].item()
        if mode == 'shared':    W = conv.W[0]
        elif mode == 'edge':    W = conv.W[e]
        elif mode == 'node':    W = conv.W[tu]
        elif mode == 'sum':     W = conv.W[tv] + conv.W[nt + e] + conv.W[tu]
        else:                   W = conv.W[(tv * ne + e) * nt + tu]
        manual[v] += x[u] @ W                          # message from u to v

    print(mode, torch.allclose(out, manual, atol=1e-5))