from abc import ABC, abstractmethod

from flax import linen as nn
from jax import numpy as jnp, Array


# pylint: disable=too-few-public-methods
class Kernel(ABC, nn.Module):
    """
    Kernel base class
    """

    @abstractmethod
    def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray):
        pass

    def __add__(self, other):
        return _Sum(self, other)

    def __mul__(self, other):
        return _Prod(self, other)


class _Sum(Kernel):
    def __init__(self, k1: Kernel, k2: Kernel):
        name = "(" + k1.name + "+" + k2.name + ")"
        super().__init__(name=name)
        self._k1 = k1
        self._k2 = k2

    def __call__(self, x1: Array, x2: Array):
        return self._k1(x1, x2) + self._k2(x1, x2)


class _Prod(Kernel):
    def __init__(self, k1: Kernel, k2: Kernel):
        name = "(" + k1.name + "*" + k2.name + ")"
        super().__init__(name=name)
        self._k1 = k1
        self._k2 = k2

    def __call__(self, x1: Array, x2: Array):
        return self._k1(x1, x2) * self._k2(x1, x2)
