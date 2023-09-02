from typing import Optional


from flax.linen import initializers
from jax import numpy as jnp, Array
from flax import linen as nn

from ramsey._src.contrib.gaussian_process.kernel.base import Kernel


class Linear(Kernel, nn.Module):
    """
    Linear Kernel
    """

    active_dims: Optional[list] = None
    sigma_b_init: Optional[initializers.Initializer] = initializers.uniform()
    sigma_v_init: Optional[initializers.Initializer] = initializers.uniform()
    offset_init: Optional[initializers.Initializer] = initializers.uniform()

    def setup(self):
        """
        Instantiates a linear covariance function

        Parameters
        ----------
        sigma_init: Optional[Initializer]
            an initializer object from Flax or None
        name: Optional[str]
            name of the layer
        """

        self._active_dims = (
            self.active_dims if isinstance(self.active_dims, list)
            else slice(self.active_dims)
        )

    @staticmethod
    def _linear(x: jnp.ndarray, y: jnp.ndarray, sigma_b, sigma_v, offset):
        x_e = x - offset
        y_e = y - offset

        x_e = jnp.expand_dims(x_e, 1)
        y_e = jnp.expand_dims(y_e, 0)

        d = jnp.sum(x_e * y_e, axis=2)
        K = sigma_v**2 * d + sigma_b**2

        return K

    @nn.compact
    def __call__(self, x1: Array, x2: Array = None):
        if x2 is None:
            x2 = x1
        dtype = x1.dtype

        log_sigma_b = self.param(
            "log_sigma_b", self.sigma_b_init, [], dtype
        )

        log_sigma_v = self.param("log_sigma_v", self.sigma_v_init, [], dtype)

        offset = self.param(
            "offset", self.offset_init, [], dtype
        )

        cov = self._linear(
            x1[..., self._active_dims],
            x2[..., self._active_dims],
            jnp.exp(log_sigma_b),
            jnp.exp(log_sigma_v),
            offset,
        )
        return cov
