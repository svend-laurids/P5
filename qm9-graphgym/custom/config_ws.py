from yacs.config import CfgNode as CN
from torch_geometric.graphgym.config import cfg as _cfg
from torch_geometric.graphgym.register import register_config


def set_cfg_ws(cfg):
    if 'weight_sharing' in cfg.gnn:      # already added, do nothing
        return
    cfg.gnn.weight_sharing = 'shared'    # shared | edge | node | sum | product
    cfg.gnn.n_node_types = 5
    cfg.gnn.n_edge_types = 4
    cfg.qm9 = CN()
    cfg.qm9.target = 4                   # 4 = HOMO-LUMO gap, 7 = U0
    cfg.qm9.subset = 0                   # >0 = only this many molecules (debugging)


register_config('weight_sharing', set_cfg_ws)   # in case GraphGym builds its config later

if 'gnn' in _cfg:                               # GraphGym's config already exists, so add now
    set_cfg_ws(_cfg)