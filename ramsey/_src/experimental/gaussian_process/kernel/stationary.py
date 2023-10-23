from typing import Optional, Union

from flax import linen as nn
from flax.linen import initializers
from jax import Array
from jax import numpy as jnp

from ramsey._src.experimental.gaussian_process.kernel.base import Kernel


# pylint: disable=invalid-name
class Periodic(Kernel, nn.Module):
    """
    Periodic covariance function.

    Attributes
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
    """

    period: float
    active_dims: Optional[list] = None
    rho_init: Optional[initializers.Initializer] = initializers.uniform()
    sigma_init: Optional[initializers.Initializer] = initializers.uniform()

    def setup(self):
        """Construct the covariance function."""
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

        log_rho = self.param("log_rho", self.rho_init, [], dtype)
        log_sigma = self.param("log_sigma", self.sigma_init, [], dtype)

        cov = periodic(
            x1[..., self._active_dims],
            x2[..., self._active_dims],
            self.period,
            jnp.exp(log_sigma),
            jnp.exp(log_rho),
        )
        return cov


class ExponentiatedQuadratic(Kernel, nn.Module):
    """
    Exponentiated quadratic covariance function.

    Attributes
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

    active_dims: Optional[list] = None
    rho_init: Optional[initializers.Initializer] = None
    sigma_init: Optional[initializers.Initializer] = None

    def setup(self):
        """Construct a stationary covariance."""
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
    x1: Array,
    x2: Array,
    sigma: float,
    rho: Union[float, jnp.ndarray],
):
    """Exponentiated-quadratic convariance function.

    Parameters
    ----------
    x1: jax.Array
        (`n x p`)-dimensional set of data points
    x2: jax.Array
        (`m x p`)-dimensional set of data points
    sigma: float
        the standard deviation of the kernel function
    rho: Union[float, np.ndarray]
        the lengthscale of the kernel function. Can be a float or a
        :math:`p`-dimensional vector if ARD-behaviour is desired

    Returns
    -------
    jax.Array
        returns a (`n x m`)-dimensional kernel matrix
    """

    def _exponentiated_quadratic(x, y, sigma, rho):
        x_e = jnp.expand_dims(x, 1) / rho
        y_e = jnp.expand_dims(y, 0) / rho
        d = jnp.sum(jnp.square(x_e - y_e), axis=2)
        K = sigma * jnp.exp(-0.5 * d)
        return K

    return _exponentiated_quadratic(x1, x2, sigma, rho)


# pylint: disable=invalid-name
def periodic(x1: Array, x2: Array, period, sigma, rho):
    """Periodic convariance function.

    Parameters
    ----------
    x1: jax.Array
        (`n x p`)-dimensional set of data points
    x2: jax.Array
        (`m x p`)-dimensional set of data points
    period: float
        the period
    sigma: float
        the standard deviation of the kernel function
    rho: Union[float, np.ndarray]
        the lengthscale of the kernel function. Can be a float or a
        :math:`p`-dimensional vector if ARD-behaviour is desired

    Returns
    -------
    jax.Array
        returns a (`n x m`)-dimensional Gram matrix
    """

    def _periodic(x, y, period, sigma, rho):
        x_e = jnp.expand_dims(x, 1)
        y_e = jnp.expand_dims(y, 0)
        r2 = jnp.sum((x_e - y_e) ** 2, axis=2)
        r = jnp.sqrt(r2)
        K = sigma * jnp.exp(-2 / rho**2 * jnp.sin(jnp.pi * r / period) ** 2)
        return K

    return _periodic(x1, x2, period, sigma, rho)
