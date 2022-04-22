
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

    return optimizer


def get_sde_loss_fn(sde, train=True, reduce_mean=True, likelihood_weighting=True, eps=1e-5):
    """
    Create a loss function for training with arbitrary SDEs.

    :param sde: An 'sde_lib.SDE` object that represents the forward SDE.
    :param train: `True`for training loss and `False`for evaluation loss
    :param reduce_mean: if `True`, average the loss across data dimensions.
                        Otherwise sum the loss across data dimensions.
    :param continuous: `True` indicates that the model is defined to take continuous time steps.
                        Otherwise it requires ad-hoc interpolation to take continuous time steps.
    :param likelihood_weighting: if `True`, weight the mixture of score matching losses according
                        to https://arxiv.org/abs/2101.09258;  otherwise use the weighting recommended in our paper.
    :param eps: A `float` number. The smallest time step to sample from.
    :return: A loss function.
    """

    def loss_fn(model, x, eps=1e-5):
        """
        The loss function for training score-based generative models
        :param model: a score model
        :param x: a mini-batch of training data
        :return: A scalar that represents the average loss value across the mini-batch
        """

        # sample t ~ U(0,1)
        random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
        z = torch.randn_like(x)

        # sample x(t) ~p_0t(x(t)|x(0)) for a random t
        # compute expected and variance of x(t) using the variation of constants solution.
        mean, std = sde.marginal_pro(x, random_t)
        # sample x(t) ~ mean + std * z ; z ~ N(0,1)
        perturbed_x = mean + std[:, None, None, None] * z

        score = model(perturbed_x, random_t)

        # weighted loss function
        loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
        return loss

    return loss_fn


def get_step_fn(sde, train, optimize_fn=None, reduce_mean=False, likelihood_weighting=True):
    """
    Create a one-step training/evaluation function
    :param sde:  An `sde_lib.SDE` object that represents the forward SDE.
    :param train:
    :param optimize_fn:   An optimization function.
    :param reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    :param likelihood_weighting: If `True`, weight the mixture of score matching losses according to
      https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.
    :return: A one-step function for training or evaluation
    """

    loss_fn = get_sde_loss_fn(sde)

    def step_fn(state, batch):
        """
        Running one step of training or evaluation
        :param state: A dictionary of training information, containing the score model, optimizer,
                      EMA status, and number of optimization steps.
        :param batch: A mini-batch of training/evaluation data.
        :return:
            loss: The average loss value of this state.
        """
        model = state['model']
        if train:
            optimizer = state['optimizer']
            optimizer.zero_grad()
            loss = loss_fn(model, batch)
            loss.backward()
            optimize_fn(optimizer, model.parameters(), step=state['step'])
            state['step'] += 1
            state['ema'].update(model.parameters())
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch)
                ema.restore(model.parameters())

        return loss

    return step_fn