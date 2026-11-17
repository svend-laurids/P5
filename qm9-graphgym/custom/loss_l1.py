import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss


@register_loss('l1_loss')
def l1_loss(pred, true):
    if cfg.model.loss_fun == 'l1':
        loss = F.l1_loss(pred, true.float(), reduction=cfg.model.size_average)
        return loss, pred
    # returning None for any other loss lets GraphGym use its built-in ones