from typing import Optional

import haiku as hk
from jax import numpy as jnp

from ramsey._src.gaussian_process.kernel.base import Kernel


class Linear(Kernel):
    """
    Linear Kernel
    """

    def __init__(
        self,
        active_dims: Optional[list] = None,
        sigma_b_init: Optional[hk.initializers.Initializer] = None,
        sigma_v_init: Optional[hk.initializers.Initializer] = None,
        offset_init: Optional[hk.initializers.Initializer] = None,
        name: Optional[str] = None,
    ):
        """
        Instantiates a linear covariance function

        Parameters
        ----------
        sigma_init: Optional[Initializer]
            an initializer object from Haiku or None
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

    def _linear(self, x: jnp.ndarray, y: jnp.ndarray, sigma_b, sigma_v, offset):
        """
        Calculates Gram matrix with linear covariance function

        For details on the linear kernel see
        https://www.cs.toronto.edu/~duvenaud/cookbook/
        """

        x_e = x - offset
        y_e = y - offset

        x_e = jnp.expand_dims(x_e, 1)
        y_e = jnp.expand_dims(y_e, 0)

        d = jnp.sum(x_e * y_e, axis=2)
        K = sigma_v**2 * d + sigma_b**2

        return K

    def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray = None):
        if x2 is None:
            x2 = x1
        dtype = x1.dtype

        sigma_b_init = self.sigma_b_init
        if sigma_b_init is None:
            sigma_b_init = hk.initializers.RandomUniform(
                jnp.log(0.1), jnp.log(1.0)
            )
        log_sigma_b = hk.get_parameter(
            "log_sigma_b", [], dtype=dtype, init=sigma_b_init
        )

        sigma_v_init = self.sigma_v_init
        if sigma_v_init is None:
            sigma_v_init = hk.initializers.RandomUniform(
                jnp.log(0.1), jnp.log(1.0)
            )
        log_sigma_v = hk.get_parameter(
            "log_sigma_v", [], dtype=dtype, init=sigma_v_init
        )

        offset_init = self.offset_init
        if offset_init is None:
            offset_init = hk.initializers.RandomUniform(-1, 1)
        offset = hk.get_parameter("offset", [], dtype=dtype, init=offset_init)

        cov = self._linear(
            x1[..., self.active_dims],
            x2[..., self.active_dims],
            jnp.exp(log_sigma_b),
            jnp.exp(log_sigma_v),
            offset,
        )
        return cov
