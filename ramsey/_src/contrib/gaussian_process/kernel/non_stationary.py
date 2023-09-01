from typing import Optional


from flax.linen import initializers
from jax import numpy as jnp, Array

from ramsey._src.contrib.gaussian_process.kernel.base import Kernel


class Linear(Kernel):
    """
    Linear Kernel
    """

    def __init__(
        self,
        active_dims: Optional[list] = None,
        sigma_b_init: Optional[initializers.Initializer] = None,
        sigma_v_init: Optional[initializers.Initializer] = None,
        offset_init: Optional[initializers.Initializer] = None,
        name: Optional[str] = None,
    ):
        """
        Instantiates a linear covariance function

        Parameters
        ----------
        sigma_init: Optional[Initializer]
            an initializer object from Flax or None
        name: Optional[str]
            name of the layer
        """

        super().__init__(name=name)

        self.active_dims = (
            active_dims if isinstance(active_dims, list) else slice(active_dims)
        )
        self.sigma_b_init = sigma_b_init
        self.sigma_v_init = sigma_v_init
        self.offset_init = offset_init

    @staticmethod
    def _linear(x: jnp.ndarray, y: jnp.ndarray, sigma_b, sigma_v, offset):
        x_e = x - offset
        y_e = y - offset

        x_e = jnp.expand_dims(x_e, 1)
        y_e = jnp.expand_dims(y_e, 0)

        d = jnp.sum(x_e * y_e, axis=2)
        K = sigma_v**2 * d + sigma_b**2

        return K

    def __call__(self, x1: Array, x2: Array = None):
        if x2 is None:
            x2 = x1
        dtype = x1.dtype

        sigma_b_init = self.sigma_b_init
        if sigma_b_init is None:
            sigma_b_init = initializers.constant(jnp.log(1.0))
        log_sigma_b = self.param(
            "log_sigma_b", sigma_b_init, [], dtype
        )

        sigma_v_init = self.sigma_v_init
        if sigma_v_init is None:
            sigma_v_init = initializers.constant(jnp.log(1.0))
        log_sigma_v = self.param("log_sigma_v", sigma_v_init, [], dtype)

        offset_init = self.offset_init
        if offset_init is None:
            offset_init = initializers.uniform()
        offset = self.param(
            "offset", offset_init, [], dtype
        )

        cov = self._linear(
            x1[..., self.active_dims],
            x2[..., self.active_dims],
            jnp.exp(log_sigma_b),
            jnp.exp(log_sigma_v),
            offset,
        )
        return cov
