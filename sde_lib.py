""" Abstract SDE classes """
import abc
import torch
import numpy as np


class SDE(abc.ABC):
    """
    SDE abstract class
    Functions are designed for a mini-batch of inputs
    """

    def __init__(self, N):
        """ Construct an SDE

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
        """Compute log-density of the prior distribution.

        Useful for computing the log-likelihood via probability flow ODE.

        Args:
          z: latent code
        Returns:
          log probability density
        """
        pass

    def discretize(self, x, t):
        """
        Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i

        Useful for reverse diffusion sampling and probability flow sampling.
        Defaults to Euler-Maruyama discretization.

        :param x: a torch tensor
        :param t: a torch float representing the time step (from 0 to `self.T`)
        :return: f, G
        """
        dt = 1/self.N
        drift, diffusion = self.sde(x,t)
        f = drift * dt
        G = diffusion * torch.sqrt(dt.clone().to(t.device))
        return f, G

    def reverse(self, score_fn, probability_flow=False):
        """
        Create the reverse-time SDE/ODE

        :param score_fn: A time-dependent score-based model that takes x and t and returns the score.
        :param probability_flow:  If `True`, create the reverse-time ODE used for probability flow sampling.
        :return: the reversed SDE
        """
        N = self.N
        T = self.T
        sde_fn = self.sde
        discretize_fn = self.discretize

        # Build the class for reverse-time SDE
        class RSDE(self.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t):
                """ Create the drift and diffsuion functions for the reverse SDE/ODE """
                drift, diffusion = sde_fn(x,t)
                score = score_fn(x,t)
                drift = drift - diffusion[:, None, None, None]**2*score * (0.5 if self.probability_flow else 1.)
                # Set the diffusion function to zero for ODES.
                diffusion = 0. if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(self, x, t):
                """ Create discretized iteration rules for the reverse diffusion sampler """
                f, G = discretize_fn(x,t)
                rev_f = f - G[:, None, None, None]**2 * score_fn(x,t) * (0.5 if self.probability_flow else 1.)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()



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
        diffusion = (self.sigma**t).clone().detach().to(self.device)
        return drift, diffusion

    def marginal_prob(self, x, t):
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
        mean = x

        return mean, std

    # REVIEW it should be multiply by the stationary std
    def prior_sampling(self, shape):
        """
        Sampling from the prior distribution
        :param shape: image shape
        :return: A PyTorch tensor 
        """
        t = torch.ones(shape[0], device=self.device) * self.T
        init_x = torch.randn(*shape)
        _, marginal_prob_std = self.marginal_prob(init_x, t)
        return init_x * marginal_prob_std[:, None, None, None]

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2. * np.log(2 * np.pi * self.sigma ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (2 * self.sigma ** 2)
