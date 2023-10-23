from abc import ABC, abstractmethod

from flax import linen as nn
from jax import Array


# pylint: disable=too-few-public-methods
class Kernel(ABC):
    """Kernel base class."""

    @abstractmethod
    def __call__(self, x1: Array, x2: Array):
        """Compute the Gram matrix induced by the covariance function.

        Parameters
        ----------
        x1: jax.Array
            (`n x p`)-dimensional set of data points
        x2: jax.Array
            (`m x p`)-dimensional set of data points

        Returns
        -------
        jax.Array
            returns (`n x m`)-dimensional set of data points
        """

    def __add__(self, other):
        """Add two kernels."""
        return _Sum(self, other)

    def __mul__(self, other):
        """Multiply two kernels."""
        return _Prod(self, other)


class _Sum(Kernel, nn.Module):
    k1: Kernel
    k2: Kernel

    @nn.compact
    def __call__(self, x1: Array, x2: Array):
        return self.k1(x1, x2) + self.k2(x1, x2)


class _Prod(Kernel, nn.Module):
    k1: Kernel
    k2: Kernel

    @nn.compact
    def __call__(self, x1: Array, x2: Array):
        return self.k1(x1, x2) * self.k2(x1, x2)
