from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config

@register_config('tgnn_config')
def set_cfg_tgnn(cfg):
    cfg.tgnn_config = CN()
    cfg.tgnn_config.num_layers = 2
    cfg.tgnn_config.has_batchnorm = True
    cfg.tgnn_config.has_activation = True
    cfg.tgnn_config.has_dropout = True
    cfg.tgnn_config.dropout = 0.0
    cfg.tgnn_config.activation = 'relu'
    cfg.tgnn_config.messagepassing = 'tgnn_linear'
    cfg.tgnn_config.dim_in = 11
    cfg.tgnn_config.dim_inner = 20 
    cfg.tgnn_config.dim_out = 3
    cfg.tgnn_config.aggregation = 'sum'
    cfg.tgnn_config.l2norm = False
    cfg.tgnn_config.stage_type = 'tgnn_stack'
    cfg.tgnn_config.batchnorm_epsilon = 0.000001
    cfg.tgnn_config.momentum = 0.1
