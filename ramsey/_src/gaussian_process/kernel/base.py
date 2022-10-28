from abc import ABC, abstractmethod

import haiku as hk
from jax import numpy as jnp


# pylint: disable=too-few-public-methods
class Kernel(ABC, hk.Module):
    """
    Kernel base class
    """

    @abstractmethod
    def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray):
        pass

    def __add__(self, other):
        class AddKernel(Kernel):
            """Class for addition of 2 kernels"""

            def __init__(self, k1: Kernel, k2: Kernel):
                name = k1.name + "_add_" + k2.name
                super().__init__(name=name)
                self._k1 = k1
                self._k2 = k2

            def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray):
                return self._k1(x1, x2) + self._k2(x1, x2)

        return AddKernel(self, other)

    def __mul__(self, other):
        class MulKernel(Kernel):
            """Class for multiplication of 2 kernels"""

            def __init__(self, k1: Kernel, k2: Kernel):
                name = k1.name + "_mul_" + k2.name
                super().__init__(name=name)
                self._k1 = k1
                self._k2 = k2

            def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray):
                return self._k1(x1, x2) * self._k2(x1, x2)

        return MulKernel(self, other)
