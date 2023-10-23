from typing import Optional

from flax import linen as nn
from flax.linen import initializers
from jax import Array
from jax import numpy as jnp

from ramsey._src.experimental.gaussian_process.kernel.base import Kernel


class Linear(Kernel, nn.Module):
    """Linear covariance function.

    Parameters
    ----------
    active_dims: Optional[list]
        the indexes of the dimensions the kernel acts upon
    sigma_b_init: Optional[initializers.Initializer]
        an initializer object from Flax or None
    sigma_v_init: Optional[initializers.Initializer]
        an initializer object from Flax or None
    offset_init: Optional[initializers.Initializer]
        an initializer object from Flax or None
    """

    active_dims: Optional[list] = None
    sigma_b_init: Optional[initializers.Initializer] = initializers.uniform()
    sigma_v_init: Optional[initializers.Initializer] = initializers.uniform()
    offset_init: Optional[initializers.Initializer] = initializers.uniform()

    def setup(self):
        """Construct parameters."""
        self._active_dims = (
            self.active_dims
            if isinstance(self.active_dims, list)
            else slice(self.active_dims)
        )

    @nn.compact
    def __call__(self, x1: Array, x2: Array = None):
        """Call the covariance function."""
        if x2 is None:
            x2 = x1
        dtype = x1.dtype

        log_sigma_b = self.param("log_sigma_b", self.sigma_b_init, [], dtype)

        log_sigma_v = self.param("log_sigma_v", self.sigma_v_init, [], dtype)

        offset = self.param("offset", self.offset_init, [], dtype)

        cov = linear(
            x1[..., self._active_dims],
            x2[..., self._active_dims],
            jnp.exp(log_sigma_b),
            jnp.exp(log_sigma_v),
            offset,
        )
        return cov


def linear(x1: Array, x2: Array, sigma_b, sigma_v, offset):
    r"""Linear convariance function.

    Parameters
    ----------
    x1: jax.Array
        :math:`n x p`-dimensional set of data points
    x2: jax.Array
        :math:`m x p`-dimensional set of data points
    sigma_b: float
        the standard deviation of the kernel function
    sigma_v: float
        the standard deviation of the kernel function
    offset: float

    Returns
    -------
    jax.Array
        returns a :math:`n x m`-dimensional Gram matrix
    """

    def _linear(x1: Array, x2: Array, sigma_b, sigma_v, offset):
        x_e = x1 - offset
        y_e = x2 - offset
        x_e = jnp.expand_dims(x_e, 1)
        y_e = jnp.expand_dims(y_e, 0)
        d = jnp.sum(x_e * y_e, axis=2)
        K = sigma_v**2 * d + sigma_b**2
        return K

    return _linear(x1, x2, sigma_b, sigma_v, offset)
