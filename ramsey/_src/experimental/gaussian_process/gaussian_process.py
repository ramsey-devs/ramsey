from typing import Optional

from flax import linen as nn
from flax.linen import initializers
from jax import Array
from jax import numpy as jnp
from jax import scipy as jsp
from numpyro import distributions as dist

from ramsey._src.experimental.gaussian_process.kernel.base import Kernel

__all__ = ["GP"]


# pylint: disable=too-many-instance-attributes,duplicate-code
class GP(nn.Module):
    """
    A Gaussian process.

    Attributes
    ----------
    kernel: Kernel
        a covariance function
    sigma_init: Optional[initializers.Initializer]
        an initializer object from Flax
    """

    kernel: Kernel
    sigma_init: Optional[initializers.Initializer] = None

    @nn.compact
    def __call__(self, x: Array, **kwargs):
        """
        Evaluate the Gaussian process.

        Parameters
        ----------
        inputs: jax.Array
            training point x
        **kwargs: keyword arguments
            Keyword arguments can include:
            - outputs: jax.Array.
            - inputs_star: jax.Array

        Returns
        -------
        numpyro.distribution
            returns a multivariate normal distribution object

        References
        ----------
        .. [1] Rasmussen, Carl E and Williams, Chris KI.
           "Gaussian Processes for Machine Learning". MIT press, 2006.
        """
        if "y" in kwargs and "x_star" in kwargs:
            return self._predictive(x, **kwargs)
        return self._marginal(x, **kwargs)

    def _get_sigma(self, dtype):
        log_sigma_init = self.sigma_init
        if log_sigma_init is None:
            log_sigma_init = initializers.constant(jnp.log(1.0))
        log_sigma = self.param("log_sigma", log_sigma_init, [], dtype)
        return log_sigma

    # pylint: disable=too-many-locals
    def _predictive(self, x: Array, y: Array, x_star: Array, jitter=10e-8):
        """Return the predictive posterior distribution.

        For details on the implemented algorithm see [1],
        Chapter 2.2 Function-space View, Algorithm, 2.1

        Parameters
        ----------
        x: jax.Array
            training point x
        y: jax.Array
            training point y
        x_star: jax.Array
            test points
        jitter: Optional[float]
            jitter to add to covariance diagonal

        Returns
        -------
        distrax.MultivariateNormalTri
            returns a multivariate normal distribution object

        References
        ----------
        .. [1] Rasmussen, Carl E and Williams, Chris KI.
            "Gaussian Processes for Machine Learning". MIT press, 2006.
        """
        log_sigma = self._get_sigma(x.dtype)

        n = x.shape[0]
        K_xx = self.kernel(x, x) + (
            jnp.square(jnp.exp(log_sigma)) + jitter
        ) * jnp.eye(n)
        K_xs_xs = self.kernel(x_star, x_star)
        K_x_xs = self.kernel(x, x_star)

        L = jnp.linalg.cholesky(K_xx)
        w = jsp.linalg.solve_triangular(L, y, lower=True)
        L_inv_K_x_xs = jsp.linalg.solve_triangular(L, K_x_xs, lower=True)

        n_star = x_star.shape[0]
        mu_star = jnp.matmul(L_inv_K_x_xs.T, w)
        cov_star = K_xs_xs - jnp.matmul(L_inv_K_x_xs.T, L_inv_K_x_xs)
        cov_star += jitter * jnp.eye(n_star)

        return dist.MultivariateNormal(
            loc=jnp.squeeze(mu_star), scale_tril=jnp.linalg.cholesky(cov_star)
        )

    def _marginal(self, x, jitter=10e-8):
        n = x.shape[0]
        log_sigma = self._get_sigma(x.dtype)
        cov = self.kernel(x, x)
        cov += (jnp.square(jnp.exp(log_sigma)) + jitter) * jnp.eye(n)
        pred_fn = dist.MultivariateNormal(
            loc=jnp.zeros(n), scale_tril=jnp.linalg.cholesky(cov)
        )
        return pred_fn
