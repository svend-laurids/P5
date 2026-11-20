import torch
import torch.nn.functional as F

from torch_geometric.graphgym.config import cfg

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.register import register_stage, register_network

from torch_geometric.nn import global_mean_pool

class TGNNLayer(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        #match cfg.tgnn_config.messagepassing:
        #    case 'linear':
        #        self.add_module('messagepassing', LinearTGNN(dim_in, dim_out))
        #    case 'bond':
        #        self.add_module('messagepassing', BondTGNN(dim_in, dim_out))
        #    case 'atom':
        #        self.add_module('messagepassing', AtomTGNN(dim_in, dim_out))
        #    case 'atombondsum':
        #        self.add_module('messagepassing', AtomBondSumTGNN(dim_in, dim_out))
        #    case _:
        #        print(f"Error: cfg.tgnn_config.messagepassing = {cfg.tgnn_config.messagepassing}, which is not a valid messenger")
        self.add_module('messagepassing', register.layer_dict[cfg.tgnn_config.messagepassing](dim_in, dim_out))
        if cfg.tgnn_config.has_batchnorm:
            self.add_module('batchnorm', torch.nn.BatchNorm1d(dim_out, eps=cfg.tgnn_config.batchnorm_epsilon, momentum=cfg.tgnn_config.momentum))
        if cfg.tgnn_config.has_dropout and cfg.tgnn_config.dropout > 0.0:
            self.add_module('dropout', torch.nn.Dropout(p=cfg.tgnn_config.dropout)) #, inplace=layer_config.mem_inplace))
        if cfg.tgnn_config.has_activation:
            self.add_module('activation', register.act_dict[cfg.tgnn_config.activation]())

    def forward(self, batch):
        batch.x = self.messagepassing(batch)

        if cfg.tgnn_config.has_batchnorm:
            batch.x = self.batchnorm(batch.x)

        if cfg.tgnn_config.has_dropout and cfg.tgnn_config.dropout > 0.0:
            batch.x = self.dropout(batch.x)

        if cfg.tgnn_config.has_activation:
            batch.x = self.activation(batch.x)

        return batch

#@register_head('readout')
#class Readout(torch.nn.Module):

@register_stage('tgnn_stack')
@register_stage('tgnn_skipsum')
@register_stage('tgnn_skipconcat')
class TGNNStage(torch.nn.Module):
    def __init__(self, dim_in, dim_out, num_layers):
        super().__init__()
        self.num_layers = num_layers
        for i in range(num_layers):
            if cfg.tgnn_config.stage_type == 'tgnn_skipconcat':
                d_in = dim_in if i == 0 else dim_in + i * dim_out
            else:
                d_in = dim_in if i == 0 else dim_out
            self.add_module(f'layer{i}', TGNNLayer(d_in, dim_out))

    def forward(self, batch):
        print(batch)

        for i, layer in enumerate(self.children()):
            x = batch.x
            batch = layer(batch)
            if cfg.tgnn_config.stage_type == 'tgnn_skipsum':
                batch.x = x + batch.x
            elif (cfg.tgnn_config.stage_type == 'tgnn_skipconcat' and i < self.num_layers - 1):
                batch.x = torch.cat([x, batch.x], dim=1)

        if cfg.tgnn_config.l2norm:
            batch.x = F.normalize(batch.x, p=2, dim=-1)

        return batch

                               
@register_network('tgnn_network')
class TGNN(torch.nn.Module):
    def __init__(self, dim_in, dim_out, **kwargs):
        super().__init__()

        TMPNN = register.stage_dict[cfg.tgnn_config.stage_type]
        #Readout = register.head_dict[cfg.tgnn_config.head]

        self.mpnn = TMPNN(dim_in=cfg.tgnn_config.dim_in, dim_out=cfg.tgnn_config.dim_inner, num_layers=cfg.tgnn_config.num_layers)
        #self.readout = Readout(dim_in=cfg.tgnn_config.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        print(batch)
        print(batch.x)
        for module in self.children(): 
            batch = module(batch)

        x = global_mean_pool(batch.x, batch.batch)

        return x, batch.y
