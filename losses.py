
import torch
import torch.optim as optim
import numpy as np
from models import utils as mutils


def get_optimizer(config, params):
    """
    Returns a flax optimer object based on config
    :param config:
    :param params:
    :return:
    """

    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')