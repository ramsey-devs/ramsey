import abc

from jax import numpy as jnp
from jax import random


# pylint: disable=too-few-public-methods
class BayesianLayer(metaclass=abc.ABCMeta):
    """
    Abstract BayesianLayer class
    """

    @abc.abstractmethod
    def __call__(
        self, x: jnp.ndarray, key: random.PRNGKey, is_training: bool = False
    ):
        pass
