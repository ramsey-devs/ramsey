import abc

from jax import numpy as jnp


class Kernel(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray):
        pass
