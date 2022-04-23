""" Abstract SDE classes """
import abc
from abc import ABC
import torch
import numpy as np


class SDE(ABC):
    """
    SDE abstract class
    Functions are designed for a mini-batch of inputs
    """

    def __init__(self, N):
        """
        Construct an SDE
        :param N: number of discretization time steps (time partition)
        """
        super().__init__()
        self.N = N

        @property
        @abc.abstractmethod
        def T(self):
            """ End time of the SDE """
            pass

        @abc.abstractmethod
        def sde(self, x, t):
            pass

        @abc.abstractmethod
        def marginal_prob(self, x, t):
            """ Parameters to determine the marginal distribution of the SDE $p_t(x)$ """
            pass

        @abc.abstractmethod
        def prior_sampling(self, shape):
            """ Generate one sample from the prior distribution $p_T(x)$"""
            pass

        @abc.abstractmethod
        def prior_logp(self, z):
            """
            Compute log-density of the prior distribution

            Useful for computing the log-likelihood via probability flow ODE.

            :param z: latent code
            :return: log probability density
            """
            pass


class BASIC_SDE(SDE):
    def __init__(self, sigma=25, N=1000, device=None):
        """
        Construct a simple SDE dx = sigma**t dW

        :param beta_min:
        :param beta_max:
        :param N: number of discretization steps
        """
        super().__init__(N)
        self.N = N
        self.sigma = sigma
        self.device = device

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        """
        The forward SDE $dx = \sigma^t dW$
        :param self:
        :param x: x init
        :param t:
        :return:
        """
        drift = 0
        diffusion = torch.tensor(self.sigma**t, device=self.device)
        return drift, diffusion

    def marginal_pro(self, x, t):
        """
        The marginal probability at x(t=T)
        Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.
        :param self:
        :param x:
        :param t:
        :return:
        """
        t = t.clone().to(self.device)
        std = torch.sqrt((self.sigma**(2 * t) - 1.) / (2. * np.log(self.sigma)))
        mean = 0

        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)
