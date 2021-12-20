import abc

import jax.numpy as np
import numpyro.distributions as dist
from chex import assert_axis_dimension
from jax.nn import softplus


# pylint: disable=too-few-public-methods
class Family(abc.ABC):
    """
    Distributional family
    """

    def __init__(self, distribution):
        self._distribution = distribution

    @abc.abstractmethod
    def __call__(
        self,
        target: np.ndarray,
        x_target: np.ndarray,
        y: np.ndarray,  # pylint: disable=invalid-name
    ):
        pass

    @staticmethod
    def _check_dims(
        mean,
        x,  # pylint: disable=invalid-name
        y,  # pylint: disable=invalid-name
    ):
        assert_axis_dimension(mean, 0, x.shape[0])
        assert_axis_dimension(mean, 1, x.shape[1])
        assert_axis_dimension(mean, 2, y.shape[2])


class Gaussian(Family):
    """
    Family of Gaussian distributions
    """

    def __init__(self):
        super().__init__(dist.Normal)

    def __call__(
        self,
        target: np.ndarray,
        x_target: np.ndarray,
        y: np.ndarray,  # pylint: disable=invalid-name
    ):
        mean, log_sigma = np.split(target, 2, axis=-1)
        sigma = 0.1 + 0.9 * softplus(log_sigma)
        super()._check_dims(mean, x_target, y)

        return dist.Normal(loc=mean, scale=sigma)


class NegativeBinomial(Family):
    """
    Family of negative binomial distributions
    """

    def __init__(self):
        super().__init__(dist.NegativeBinomial2)

    def __call__(
        self,
        target: np.ndarray,
        x_target: np.ndarray,
        y: np.ndarray,  # pylint: disable=invalid-name
    ):
        mean, concentration = np.split(target, 2, axis=-1)
        mean = np.exp(mean)
        concentration = np.exp(concentration)
        super()._check_dims(mean, x_target, y)

        return dist.NegativeBinomial2(mean=mean, concentration=concentration)
