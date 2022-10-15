from typing import Optional

import distrax
import haiku as hk
from jax import numpy as jnp

from ramsey._src.family import Family, Gaussian

__all__ = ["GP"]


# pylint: disable=too-many-instance-attributes,duplicate-code
class GP(hk.Module):
    """
    A Gaussian process

    Implements the core structure of a Gaussian process.
    """

    def __init__(
        self,
        kernel: hk.Module,
        sigma_init: Optional[hk.initializers.Initializer] = None,
        family: Family = Gaussian(),
        name: Optional[str] = None,
    ):
        """
        Instantiates a Gaussian process

        Parameters
        ----------
        kernel: hk.Module
            a covariance function object
        sigma_init: Optional[Initializer]
            an initializer object from Haiku or None
        family: Family
            an exponential family object specifying the distribution of the data
            and which likelihood is going to be used
        name: Optional[str]
            name of the layer
        """

        super().__init__(name=name)
        self._kernel = kernel
        self._family = family
        self.sigma_init = sigma_init

    def __call__(self, x: jnp.ndarray, **kwargs):
        if "y" in kwargs and "x_star" in kwargs:
            return self._predictive(x, **kwargs)
        return self._marginal(x)

    def _get_sigma(self, dtype):
        log_sigma_init = self.sigma_init
        if log_sigma_init is None:
            log_sigma_init = hk.initializers.RandomUniform(
                jnp.log(0.1), jnp.log(1.0)
            )
        log_sigma = hk.get_parameter(
            "log_sigma", [], dtype=dtype, init=log_sigma_init
        )
        return log_sigma

    def _predictive(
        self, x: jnp.ndarray, y: jnp.ndarray, x_star: jnp.ndarray, jitter=10e-8
    ):
        n = x.shape[0]
        log_sigma = self._get_sigma(x.dtype)

        K_xx = self._kernel(x, x) + (jnp.exp(log_sigma) + jitter) * jnp.eye(n)
        K_x_xs = self._kernel(x, x_star)
        K_xs_xs = self._kernel(x_star, x_star)
        K_xx_inv = jnp.linalg.inv(K_xx)

        mu_star = K_x_xs.T @ K_xx_inv @ y
        cov_star = K_xs_xs - K_x_xs.T @ K_xx_inv @ K_x_xs

        return distrax.MultivariateNormalTri(
            jnp.squeeze(mu_star), jnp.linalg.cholesky(cov_star)
        )

    def _marginal(self, x, jitter=10e-8):
        n = x.shape[0]

        log_sigma = self._get_sigma(x.dtype)
        cov = self._kernel(x, x)
        cov += (jnp.exp(log_sigma) + jitter) * jnp.eye(n)

        return distrax.MultivariateNormalTri(
            jnp.zeros(n), jnp.linalg.cholesky(cov)
        )
