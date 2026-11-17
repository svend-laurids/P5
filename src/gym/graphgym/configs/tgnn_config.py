from yacs.config import CfgNode as CN
from graphgym.register import register_config

def set_cfg_tgnn(cfg):
    cfg.tgnn = CN()
    cfg.tgnn.n_layers = 2
    cfg.tgnn.act = 'relu'
    cfg.tgnn.dropout = 0.0
    cfg.tgnn.agg = 'add'
    cfg.tgnn.msg = 'linear'
    cfg.tgnn.l2norm = False
    cfg.tgnn.dim_inner = 20

register_config("tgnn", set_cfg_tgnn)
