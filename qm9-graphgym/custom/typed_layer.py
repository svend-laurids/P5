import torch, torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_layer
from torch_geometric.nn import MessagePassing

class TypedConv(MessagePassing):
    def __init__(self, d_in, d_out, mode, n_nt=5, n_et=4):
        super().__init__(aggr=cfg.gnn.agg, node_dim=0)
        self.mode, self.n_nt, self.n_et = mode, n_nt, n_et
        K = {'shared': 1, 'edge': n_et, 'node': n_nt,
             'sum': n_nt + n_et, 'product': n_nt * n_et * n_nt}[mode]
        self.W = nn.Parameter(torch.empty(K, d_in, d_out))
        nn.init.xavier_uniform_(self.W)
        self.W0 = nn.Linear(d_in, d_out)           # self term, eq. 2.10

    def forward(self, x, edge_index, node_type, edge_type):
        xw = torch.einsum('nd,kdo->nko', x, self.W)
        return self.propagate(edge_index, xw=xw, nt=node_type, et=edge_type) + self.W0(x)

    def message(self, xw_j, nt_i, nt_j, et):
        e = torch.arange(xw_j.size(0), device=xw_j.device)
        if self.mode == 'shared':  return xw_j[e, 0]
        if self.mode == 'edge':    return xw_j[e, et]
        if self.mode == 'node':    return xw_j[e, nt_j]
        if self.mode == 'sum':     # layout: [5 node matrices | 4 edge matrices]
            return xw_j[e, nt_i] + xw_j[e, self.n_nt + et] + xw_j[e, nt_j]
        k = (nt_i * self.n_et + et) * self.n_nt + nt_j
        return xw_j[e, k]

@register_layer('typedconv')
class TypedConvLayer(nn.Module):
    def __init__(self, layer_config, **kwargs):
        super().__init__()
        self.model = TypedConv(layer_config.dim_in, layer_config.dim_out,
                               cfg.gnn.weight_sharing)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index,
                             batch.node_type, batch.edge_type)
        return batch