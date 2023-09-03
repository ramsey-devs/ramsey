from typing import Optional, Union

from flax import linen as nn
from flax.linen import initializers
from jax import numpy as jnp, Array

from ramsey._src.contrib.gaussian_process.kernel.base import Kernel


# pylint: disable=invalid-name
class Periodic(Kernel, nn.Module):
    """
    Periodic Kernel / Exp-Sine-Squared Kernel
    """

    period: float
    active_dims: Optional[list] = None
    rho_init: Optional[initializers.Initializer] = initializers.uniform()
    sigma_init: Optional[initializers.Initializer] = initializers.uniform()

    def setup(self):
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

        self._active_dims = (
            self.active_dims
            if isinstance(self.active_dims, list)
            else slice(self.active_dims)
        )

    @nn.compact
    def __call__(self, x1: Array, x2: Array = None):
        if x2 is None:
            x2 = x1
        dtype = x1.dtype

        log_rho = self.param("log_rho", self.rho_init, [], dtype)
        log_sigma = self.param("log_sigma", self.sigma_init, [], dtype)

        cov = self._periodic(
            x1[..., self._active_dims],
            x2[..., self._active_dims],
            jnp.exp(log_sigma),
            jnp.exp(log_rho),
        )
        return cov

    def _periodic(self, x: Array, y: Array, rho, sigma):
        """
        Calculates Gram matrix with periodic covariance function

        For details on the periodic kernel see chp. 18.2.1.4 in [1]

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


class ExponentiatedQuadratic(Kernel, nn.Module):
    """
    Exponentiated quadratic covariance function
    """

    active_dims: Optional[list] = None
    rho_init: Optional[initializers.Initializer] = None
    sigma_init: Optional[initializers.Initializer] = None

    def setup(self):
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

        self._active_dims = (
            self.active_dims if isinstance(self.active_dims, list) else slice(self.active_dims)
        )

    @nn.compact
    def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray = None):
        if x2 is None:
            x2 = x1
        dtype = x1.dtype

        rho_init = self.rho_init
        if rho_init is None:
            rho_init = initializers.constant(jnp.log(1.0))
        log_rho = self.param("log_rho", rho_init, [], dtype)

        sigma_init = self.sigma_init
        if sigma_init is None:
            sigma_init = initializers.constant(jnp.log(1.0))
        log_sigma = self.param("log_sigma", sigma_init, [], dtype)

        cov = exponentiated_quadratic(
            x1[..., self._active_dims],
            x2[..., self._active_dims],
            jnp.square(jnp.exp(log_sigma)),
            jnp.exp(log_rho),
        )
        return cov


# pylint: disable=invalid-name
def exponentiated_quadratic(
    x: Array,
    y: Array,
    sigma=1.0,
    rho: Union[float, jnp.ndarray] = 1.0,
):
    """
    Exponentiated-quadratic convariance function

    Computes the cross-kernel between two-sets of data points X and Y

    Parameters
    -----------
    x: jax.Array
        (`n x p`)-dimensional set of data points
    y: jax.Array
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
