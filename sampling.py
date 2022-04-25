import numpy as np
import functools
import abc

import torch
from models import utils as mutils

_CORRECTORS = {}
_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """ A decorator for registering predictor classes """
    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def register_corrector(cls=None, *, name=None):
    """A decorator for registering corrector classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _CORRECTORS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _CORRECTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    return _PREDICTORS[name]


def get_corrector(name):
    return _CORRECTORS[name]


def get_sampling_fn(config, sde, shape, inverse_scaler, eps):
    """
    Create a sampling function

    :param config: A `ml_collections.ConfigDict` object that contains all configuration information.
    :param sde: A `sde_lib.SDE` object that represents the forward SDE.
    :param shape: A sequence of integers representing the expected shape of a single sample.
    :param inverse_scaler: The inverse data normalizer function.
    :param eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

    :return: A function that takes random states and a replicated training state and outputs samples with the
      training dimensions matching `shape`.
    """

    sampler_name = config.sampling.method

    # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
    if sampler_name.lower == 'pc':
        predictor = get_predictor(config.sampling.predictor.lower())
        corrector = get_corrector(config.sampling.corrector.lower())
        sampling_fn = get_pc_sampler(sde=sde,
                                     shape=shape,
                                     predictor=predictor,
                                     corrector=corrector,
                                     inverse_scaler=inverse_scaler,
                                     snr=config.sampling.snr,
                                     n_steps=config.sampling.n_steps_each,
                                     probability_flow=config.sampling.probability_flow,
                                     continuous=config.training.continuous,
                                     denoise=config.sampling.noise_removal,
                                     eps=eps,
                                     device=config.device)
    else:
        raise ValueError(f"Sampler name {sampler_name} unknown.")

    return sampling_fn







class Predictor(abc.ABC):
    """ The abstract class for a predictor algorithm """

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t):
        """
        One update of the predictor
        :param x: A Pytorch tensor representing the current state
        :param t: A Pytorch tensor representing the current time step

        :return:
               x: A Pytorch tensor of the next state
               x_mean: A Pytorch tensor. The next state without random noise. Useful for denoising.
        """
        pass

class Corrector(abc.ABC):
    """ The abstract class for a corrector algorithm """

    def __init__(self, sde, score_fn, snr, n_steps):
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t):
        """
        One update of the corrector

        :param x: A PyTorch tensor representing the current state
        :param t: A PyTorch tensor representing the current time step
        :return:
            x: A PyTorch tensor representing the next state
            x_mean: A PyTorch tensor representing the next state without random noise. Useful for denoising
        """
        pass

@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t):
        dt = -1. / self.rsde.N
        z = torch.randn_like(x)
        drift, diffusion = self.rsde.sde(x,t)
        x_mean = x + drift*dt
        x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt)*z
        return x, x_mean


@register_predictor(name='none')
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, sde, score_fn, probability_flow=False):
        pass

    def update_fn(self, x, t):
        return x, x

@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):

def shared_predictor_update_fn(x, t, sde, model, predictor, probability_flow, continuous):
    """A wrapper that configures and returns the update function of predictors."""
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    if predictor is None:
        # Corrector-only sampler
        predictor_obj = NonePredictor(sde, score_fn, probability_flow)
    else:
        predictor_obj = predictor(sde, score_fn, probability_flow)
    return predictor_obj.update_fn(x, t)


def shared_corrector_update_fn(x, t, sde, model, corrector, continuous, snr, n_steps):
    """A wrapper tha configures and returns the update function of correctors."""
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    if corrector is None:
        # Predictor-only sampler
        corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
    else:
        corrector_obj = corrector(sde, score_fn, snr, n_steps)
    return corrector_obj.update_fn(x, t)


def get_pc_sampler(sde, shape, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda'):
    """
    Create a Predictor-Corrector (PC) sampler

    :param sde: An `sde_lib.SDE` object representing the forward SDE.
    :param shape: A sequence of integers. The expected shape of a single sample.
    :param predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    :param corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    :param inverse_scaler: The inverse data normalizer.
    :param snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    :param n_steps: An integer. The number of corrector steps per predictor update.
    :param probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    :param continuous: `True` indicates that the score model was continuously trained.
    :param denoise: If `True`, add one-step denoising to the final samples.
    :param eps:  A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
    :param device: PyTorch device.

    :return: A sampling function that returns samples and the number of function evaluations during sampling.
    """

    # Create predictor & corrector update functions
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)

    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    def pc_sampler(model):
        """
        The PC sampler function

        :param model: a score model
        :return: samples, number of function evaluations
        """
        with torch.no_grad():
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            # define the reverse time partition from [T,0)
            timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

            for i in range(sde.N):
                t = timesteps[i]
                vec_t = torch.ones(shape[0], device=t.device) * t
                x, x_mean = corrector_update_fn(x, vec_t, model=model)
                x, x_mean = predictor_update_fn(x, vec_t, model=model)

            return inverse_scaler(x_mean if denoise else x), sde.N * (n_steps +1)

    return pc_sampler