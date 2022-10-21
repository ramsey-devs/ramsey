from typing import Optional

import distrax
import haiku as hk
from jax import numpy as jnp

from ramsey._src.family import Family, Gaussian

__all__ = ["GP"]

# pylint: disable=too-many-instance-attributes,duplicate-code,too-few-public-methods
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

    # pylint: disable=too-many-locals
    def _predictive(
        self, x: jnp.ndarray, y: jnp.ndarray, x_star: jnp.ndarray, jitter=10e-8
    ):
        """
        Returns Predictive Posterior Distribution

        For details on the implemented algrothim see

            - C. E. Rasmussen & C. K. I. Williams
              Gaussian Processes for Machine Learning, 2006
              Chapter 2.2 Function-space View, Alogrithm 2.1

        Parameters
        ----------
        x: jnp.ndarray
            training point x
        y: jnp.ndarray
            training point y
        x_star: jnp.ndarray
            test points
        jitter: Optional[float]
            jitter to add to covariance diagonal

        Returns
        -------
        distrax.MultivariateNormalTri
            returns a multivariate normal distribution object
        """

        log_sigma = self._get_sigma(x.dtype)

        n = x.shape[0]
        K_xx = self._kernel(x, x) + (
            jnp.exp(log_sigma) ** 2 + jitter
        ) * jnp.eye(n)
        K_xs_xs = self._kernel(x_star, x_star)
        K_x_xs = self._kernel(x, x_star)
        K_xs_x = K_x_xs.T

        L = jnp.linalg.cholesky(K_xx)
        L_inv = jnp.linalg.inv(L)
        alpha = jnp.linalg.inv(L.T) @ (L_inv @ y)

        v = L_inv @ K_x_xs

        mu_star = K_xs_x @ alpha

        cov_star = K_xs_xs - v.T @ v

        m = x_star.shape[0]
        cov_star += jitter * jnp.eye(m)

        return distrax.MultivariateNormalTri(
            jnp.squeeze(mu_star), jnp.linalg.cholesky(cov_star)
        )

    def _marginal(self, x, jitter=10e-8):
        n = x.shape[0]

        log_sigma = self._get_sigma(x.dtype)
        cov = self._kernel(x, x)
        cov += (jnp.exp(log_sigma) ** 2 + jitter) * jnp.eye(n)

        return distrax.MultivariateNormalTri(
            jnp.zeros(n), jnp.linalg.cholesky(cov)
        )
