from typing import Optional

import distrax
import haiku as hk
from jax import numpy as jnp
from jax import scipy as jsp

__all__ = ["SparseGP"]


# pylint: disable=too-many-instance-attributes,duplicate-code
class SparseGP(hk.Module):
    """
    A sparse Gaussian process

    Implements the core structure of a sparse Gaussian process.

    References
    ----------
    [1] Titsias, Michalis K.
        "Variational Learning of Inducing Variables in Sparse Gaussian
        Processes". AISTATS, 2009
    """

    def __init__(
        self,
        kernel: hk.Module,
        m: int,
        jitter: Optional[float] = 10e-8,
        sigma_init: Optional[hk.initializers.Initializer] = None,
        x_m_init: Optional[hk.initializers.Initializer] = None,
        name: Optional[str] = None,
    ):
        """
        Instantiates a sparse Gaussian process

        Parameters
        ----------
        kernel: hk.Module
            a covariance function object
        m: Optional[int]
            number of inducing points
        jitter: Optional[float]
            additive jitter on covariance matrices diagonals to
            stabilize them against loosing positive definite property
        sigma_init: Optional[Initializer]
            an initializer object from Haiku or None
        name: Optional[str]
            name of the layer
        """

        super().__init__(name=name)
        self._kernel = kernel
        self._m = m
        self._jitter = jitter
        self._sigma_init = sigma_init
        self._x_m_init = x_m_init

    def __call__(self, x: jnp.ndarray, **kwargs):
        if "y" in kwargs and "x_star" in kwargs:
            return self._predictive(x, **kwargs)
        return self._marginal(x, **kwargs)

    def _get_sigma(self, dtype):
        log_sigma_init = self._sigma_init
        if log_sigma_init is None:
            log_sigma_init = hk.initializers.RandomUniform(
                jnp.log(0.1), jnp.log(1.0)
            )
        log_sigma = hk.get_parameter(
            "log_sigma", [], dtype=dtype, init=log_sigma_init
        )
        return log_sigma

    def _get_x_m(self, x_n: jnp.ndarray):
        """
        Returns the m inducing points x_m

        Parameters
        ----------
        x_n: jnp.ndarray
            training points x_n for initialization of inducing points x_m

        Returns
        -------
        x_m: jnp.ndarray
            inducing points x_m
        """

        d = x_n.shape[1]
        shape_x_m = (self._m, d)
        if self._x_m_init is None:
            self._x_m_init = hk.initializers.RandomUniform(
                jnp.min(x_n), jnp.max(x_n)
            )

        x_m = hk.get_parameter(
            "x_m", shape=shape_x_m, dtype=x_n.dtype, init=self._x_m_init
        )
        return x_m

    # pylint: disable=too-many-locals
    def _predictive(self, x: jnp.ndarray, y: jnp.ndarray, x_star: jnp.ndarray):
        """
        Returns the approx. Predictive Posterior Distribution

        The distribution is calculated according equation (6) in [1]

        Parameters
        ----------
        x: jnp.ndarray
            training point x
        y: jnp.ndarray
            training point y
        x_star: jnp.ndarray
            test points

        Returns
        -------
        distrax.MultivariateNormalTri
            returns a multivariate normal distribution object
        """

        log_sigma = self._get_sigma(x.dtype)
        sigma_square = jnp.square(jnp.exp(log_sigma))
        x_m = self._get_x_m(x_n=x)

        # add jitter to diagonal to increase chances that K_mm is pos. def.
        K_mm = self._kernel(x_m, x_m) + self._jitter * jnp.eye(self._m)
        K_mn = self._kernel(x_m, x)
        C = K_mm + 1 / sigma_square * (K_mn @ K_mn.T)

        h0 = self._solve_linear(C, K_mn)  # C_inv @ K_mn
        mu = 1 / sigma_square * K_mm @ h0 @ y

        K_ss = self._kernel(x_star, x_star)
        K_sm = self._kernel(x_star, x_m)
        K_ms = K_sm.T

        h1 = self._solve_linear(K_mm, mu)  # K_mm_inv @ muc
        mu_star = K_sm @ h1
        A = self._calculate_quadratic_form(C, K_mm)  # K_mm.T @ C_inv @ K_mm
        B_inv = self._calculate_quadratic_form(A, K_mm)  # K_mm * A_inv * K_mm
        # K_sm @ K_mm_inv @ K_ms
        h2 = self._calculate_quadratic_form(K_mm, K_ms)
        # K_sm @ B @ K_ms
        h3 = self._calculate_quadratic_form(B_inv, K_ms)
        cov_star = K_ss - h2 + h3 + self._jitter * jnp.eye(x_star.shape[0])

        return distrax.MultivariateNormalTri(
            jnp.squeeze(mu_star), jnp.linalg.cholesky(cov_star)
        )

    def _marginal(self, x: jnp.ndarray, y: jnp.ndarray):
        """
        Returns the variational lower bound of true log marginal likelihood.
        This quantity can be used as an objective to find the kernel
        hyperparameters and the location of the m inducing points x_m.

        The calculations are implemented according equation (9) in [1].

        Parameters
        ----------
        x: jnp.ndarray
            training point x
        y: jnp.ndarray
            training point y

        Returns
        -------
        float
             variational lower bound of true log marginal likelihood
        """

        n = x.shape[0]
        log_sigma = self._get_sigma(x.dtype)
        sigma_square = jnp.exp(2 * log_sigma)
        x_m = self._get_x_m(x_n=x)

        # add jitter to diagonal to increase chances that K_mm is pos. def.
        K_mm = self._kernel(x_m, x_m) + self._jitter * jnp.eye(self._m)
        K_mn = self._kernel(x_m, x)

        Q_nn = self._calculate_quadratic_form(
            K_mm, K_mn
        )  # K_mn.T @ K_mm_inv @ K_mn

        cov = Q_nn + ((self._jitter + sigma_square) * jnp.eye(n))

        mvn = distrax.MultivariateNormalTri(
            jnp.zeros(n), jnp.linalg.cholesky(cov)
        )

        K_nn = self._kernel(x, x)
        regularizer = -1 / (2 * sigma_square) * jnp.trace(K_nn - Q_nn)

        mll = jnp.sum(mvn.log_prob(y.T))

        variational_lower_bound = mll + regularizer

        return variational_lower_bound

    @staticmethod
    def _solve_linear(A: jnp.ndarray, b: jnp.ndarray):
        """
        If A is symmetric and positive definite then Ax=b can be solved
        by using Cholesky decomposition.

        (1) L = cholesky(A)
        (2) L*y = b
        (3) L.T * x = y

        As L is a triangular (2) can be efficiently solved for y
        and (3) can be efficiently solved for x.

        Parameters
        ----------
        A: jnp.ndarray
            A in term Ax=b
        b: jnp.ndarray
            b in term Ax=b

        Returns
        --------
        float
            x from term Ax=b
        """

        L = jnp.linalg.cholesky(A)
        y = jsp.linalg.solve_triangular(L, b, lower=True)
        x = jsp.linalg.solve_triangular(L.T, y, lower=False)

        return x

    # pylint: disable=line-too-long
    @staticmethod
    def _calculate_quadratic_form(A: jnp.ndarray, x: jnp.ndarray):
        """
        Calculates a quadratic form

            y = x.T * inv(A) * x

        using the Cholesky decomposition of A without actually computing inv(A)
        Note that A has to be symmetric and positive definite.

        Parameters
        ----------
        A: jnp.ndarray
            A in term y = x.T * inv(A) * x
        x: jnp.ndarray
            x in term y = x.T * inv(A) * x

        Returns
        -------
        float
            x.T * inv(A) * x
        """

        L = jnp.linalg.cholesky(A)
        z = jsp.linalg.solve_triangular(L, x, lower=True)
        y = z.T @ z

        return y
