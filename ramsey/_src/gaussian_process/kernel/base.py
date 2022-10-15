import abc

from jax import numpy as jnp


# pylint: disable=too-few-public-methods
class Kernel(metaclass=abc.ABCMeta):
    """
    Abstract covariance function class
    """

    @abc.abstractmethod
    def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray):
        pass
