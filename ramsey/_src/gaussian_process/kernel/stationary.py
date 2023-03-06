from typing import Optional, Union

import haiku as hk
from jax import numpy as jnp

from ramsey._src.gaussian_process.kernel.base import Kernel


# pylint: disable=invalid-name
class Periodic(Kernel):
    """
    Periodic Kernel / Exp-Sine-Squared Kernel
    """

    def __init__(
        self,
        period: float,
        active_dims: Optional[list] = None,
        rho_init: Optional[hk.initializers.Initializer] = None,
        sigma_init: Optional[hk.initializers.Initializer] = None,
        name: Optional[str] = None,
    ):
        """
        Instantiates a periodic covariance function

        Parameters
        ----------
        period: float
            the period of the periodic kernel
        active_dims: Optional[list]
            either None or a list of integers. Specified the dimensions of the
            data on which the kernel operates on
        rho_init: Optional[Initializer]
            an initializer object from Haiku or None
        sigma_init: Optional[Initializer]
            an initializer object from Haiku or None
        name: Optional[str]
            name of the layer
        """
        super().__init__(name=name)
        self.period = period
        self.active_dims = (
            active_dims if isinstance(active_dims, list) else slice(active_dims)
        )
        self.rho_init = rho_init
        self.sigma_init = sigma_init

    def _periodic(self, x: jnp.ndarray, y: jnp.ndarray, rho, sigma):
        """
        Calculates Gram matrix with periodic covariance function

        For details on the periodic kernel see [1]  18.2.1.4 Periodic Kernels

        References
        ----------
        .. [1] Kevin P. Murphy
           "Probabilistic Machine Learning: Advanced Topics" MIT Press, 2023.
        """

        x_e = jnp.expand_dims(x, 1)
        y_e = jnp.expand_dims(y, 0)

        r2 = jnp.sum((x_e - y_e) ** 2, axis=2)
        r = jnp.sqrt(r2)

        K = sigma * jnp.exp(
            -2 / rho**2 * jnp.sin(jnp.pi * r / self.period) ** 2
        )
        return K

    def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray = None):
        if x2 is None:
            x2 = x1
        dtype = x1.dtype

        rho_init = self.rho_init
        if rho_init is None:
            rho_init = hk.initializers.RandomUniform(jnp.log(0.1), jnp.log(1.0))
        log_rho = hk.get_parameter("log_rho", [], dtype=dtype, init=rho_init)

        sigma_init = self.sigma_init
        if sigma_init is None:
            sigma_init = hk.initializers.RandomUniform(
                jnp.log(0.1), jnp.log(1.0)
            )
        log_sigma = hk.get_parameter(
            "log_sigma", [], dtype=dtype, init=sigma_init
        )

        cov = self._periodic(
            x1[..., self.active_dims],
            x2[..., self.active_dims],
            jnp.exp(log_sigma),
            jnp.exp(log_rho),
        )
        return cov


class ExponentiatedQuadratic(Kernel):
    """
    Exponentiated quadratic covariance function
    """

    def __init__(
        self,
        active_dims: Optional[list] = None,
        rho_init: Optional[hk.initializers.Initializer] = None,
        sigma_init: Optional[hk.initializers.Initializer] = None,
        name: Optional[str] = None,
    ):
        """
        Instantiates the covariance function

        Parameters
        ----------
        active_dims: Optional[list]
            either None or a list of integers. Specified the dimensions of the
            data on which the kernel operates on
        rho_init: Optional[Initializer]
            an initializer object from Haiku or None
        sigma_init: Optional[Initializer]
            an initializer object from Haiku or None
        name: Optional[str]
            name of the layer
        """

        super().__init__(name=name)
        self.active_dims = (
            active_dims if isinstance(active_dims, list) else slice(active_dims)
        )
        self.rho_init = rho_init
        self.sigma_init = sigma_init

    def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray = None):
        if x2 is None:
            x2 = x1
        dtype = x1.dtype

        rho_init = self.rho_init
        if rho_init is None:
            rho_init = hk.initializers.RandomUniform(jnp.log(0.1), jnp.log(1.0))
        log_rho = hk.get_parameter("log_rho", [], dtype=dtype, init=rho_init)

        sigma_init = self.sigma_init
        if sigma_init is None:
            sigma_init = hk.initializers.RandomUniform(
                jnp.log(0.1), jnp.log(1.0)
            )
        log_sigma = hk.get_parameter(
            "log_sigma", [], dtype=dtype, init=sigma_init
        )

        cov = exponentiated_quadratic(
            x1[..., self.active_dims],
            x2[..., self.active_dims],
            jnp.square(jnp.exp(log_sigma)),
            jnp.exp(log_rho),
        )
        return cov


# pylint: disable=invalid-name
def exponentiated_quadratic(
    x: jnp.ndarray,
    y: jnp.ndarray,
    sigma=1.0,
    rho: Union[float, jnp.ndarray] = 1.0,
):
    """
    Exponentiated-quadratic convariance function

    Computes the cross-kernel between two-sets of data points X and Y

    Parameters
    -----------
    x: np.ndarray
        (`n x p`)-dimensional set of data points
    y: np.ndarray
        (`m x p`)-dimensional set of data points
    sigma: float
        the standard deviation of the kernel function
    rho: Union[float, np.ndarray]
        the lengthscale of the kernel function. Can be a float or a
        :math:`p`-dimensional vector if ARD-behaviour is desired

    Returns
    -------
    np.ndarray
        returns a (`n x m`)-dimensional kernel matrix
    """

    def _exponentiated_quadratic(x, y, sigma, rho):
        x_e = jnp.expand_dims(x, 1) / rho
        y_e = jnp.expand_dims(y, 0) / rho
        d = jnp.sum(jnp.square(x_e - y_e), axis=2)
        K = sigma * jnp.exp(-0.5 * d)
        return K

    return _exponentiated_quadratic(x, y, sigma, rho)
