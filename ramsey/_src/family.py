import abc

import jax.numpy as np
import numpyro.distributions as dist
from jax.nn import softplus


# pylint: disable=too-few-public-methods
class Family(abc.ABC):
    """
    Distributional family
    """

    def __init__(self, distribution):
        self._distribution = distribution

    @abc.abstractmethod
    def __call__(self, target: np.ndarray):
        pass


class Gaussian(Family):
    """
    Family of Gaussian distributions
    """

    def __init__(self):
        super().__init__(dist.Normal)

    def __call__(self, target: np.ndarray):
        mean, log_sigma = np.split(target, 2, axis=-1)
        sigma = 0.1 + 0.9 * softplus(log_sigma)
        return dist.Normal(loc=mean, scale=sigma)


class NegativeBinomial(Family):
    """
    Family of negative binomial distributions
    """

    def __init__(self):
        super().__init__(dist.NegativeBinomial2)

    def __call__(self, target: np.ndarray):
        mean, concentration = np.split(target, 2, axis=-1)
        mean = np.exp(mean)
        concentration = np.exp(concentration)

        return dist.NegativeBinomial2(mean=mean, concentration=concentration)
