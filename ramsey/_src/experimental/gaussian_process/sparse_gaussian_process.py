from typing import Optional

from flax import linen as nn
from flax.linen import initializers
from jax import Array
from jax import numpy as jnp
from jax import scipy as jsp
from numpyro import distributions as dist

from ramsey._src.experimental.gaussian_process.kernel.base import Kernel

__all__ = ["SparseGP"]


# pylint: disable=too-many-instance-attributes,duplicate-code
class SparseGP(nn.Module):
    """A sparse Gaussian process.

    Attributes
    ----------
    kernel: Kernel
        a covariance function
    n_inducing: int
        number of inducing points
    jitter: float
        jitter to add to the covariance matrix diagonal
    log_sigma_init: Optional[initializers.Initializer]
        an initializer object from Flax
    inducing_init: Optional[initializers.Initializer]
        an initializer object from Flax

    References
    ----------
    [1] Titsias, Michalis K.
        "Variational Learning of Inducing Variables in Sparse Gaussian
        Processes". AISTATS, 2009
    """

    kernel: Kernel
    n_inducing: int
    jitter: Optional[float] = 10e-8
    log_sigma_init: Optional[initializers.Initializer] = initializers.constant(
        jnp.log(1.0)
    )
    inducing_init: Optional[initializers.Initializer] = initializers.uniform(1)

    @nn.compact
    def __call__(self, x: Array, **kwargs):
        """Call the sparse GP."""
        if "y" in kwargs and "x_star" in kwargs:
            return self._predictive(x, **kwargs)
        return self._marginal(x, **kwargs)

    def _get_sigma(self, dtype):
        log_sigma = self.param("log_sigma", self.log_sigma_init, [], dtype)
        return log_sigma

    def _get_x_m(self, x_n: Array):
        """Create inducing points.

        Parameters
        ----------
        x_n: jax.Array
            training points x_n for initialization of inducing points x_m

        Returns
        -------
        x_m: jax.Array
            inducing points x_m
        """
        d = x_n.shape[1]
        shape_x_inducing = (self.n_inducing, d)
        x_inducing = self.param(
            "x_inducing", self.inducing_init, shape_x_inducing, x_n.dtype
        )
        return x_inducing

    # pylint: disable=too-many-locals
    def _predictive(self, x: Array, y: Array, x_star: Array):
        """Compute the approx. predictive posterior distribution.

        The distribution is calculated according equation (6) in [1]

        Parameters
        ----------
        x: jax.Array
            training point x
        y: jax.Array
            training point y
        x_star: jax.Array
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
        K_mm = self.kernel(x_m, x_m) + self.jitter * jnp.eye(self.n_inducing)
        K_mn = self.kernel(x_m, x)
        C = K_mm + 1 / sigma_square * (K_mn @ K_mn.T)

        h0 = self._solve_linear(C, K_mn)  # C_inv @ K_mn
        mu = 1 / sigma_square * K_mm @ h0 @ y

        K_ss = self.kernel(x_star, x_star)
        K_sm = self.kernel(x_star, x_m)
        K_ms = K_sm.T

        h1 = self._solve_linear(K_mm, mu)  # K_mm_inv @ muc
        mu_star = K_sm @ h1
        A = self._calculate_quadratic_form(C, K_mm)  # K_mm.T @ C_inv @ K_mm
        B_inv = self._calculate_quadratic_form(A, K_mm)  # K_mm * A_inv * K_mm
        # K_sm @ K_mm_inv @ K_ms
        h2 = self._calculate_quadratic_form(K_mm, K_ms)
        # K_sm @ B @ K_ms
        h3 = self._calculate_quadratic_form(B_inv, K_ms)
        cov_star = K_ss - h2 + h3 + self.jitter * jnp.eye(x_star.shape[0])

        return dist.MultivariateNormal(
            loc=jnp.squeeze(mu_star), scale_tril=jnp.linalg.cholesky(cov_star)
        )

    def _marginal(self, x: Array, y: Array):
        """Compute variational lower bound.

        Returns the variational lower bound of true log marginal likelihood.
        This quantity can be used as an objective to find the kernel
        hyperparameters and the location of the m inducing points x_inducing.

        The calculations are implemented according equation (9) in [1].

        Parameters
        ----------
        x: jax.Array
            training point x
        y: jax.Array
            training point y

        Returns
        -------
        float
             returns variational lower bound of true log marginal likelihood
        """
        n = x.shape[0]
        log_sigma = self._get_sigma(x.dtype)
        sigma_square = jnp.exp(2 * log_sigma)
        x_m = self._get_x_m(x_n=x)

        K_mm = self.kernel(x_m, x_m) + self.jitter * jnp.eye(self.n_inducing)
        K_mn = self.kernel(x_m, x)

        Q_nn = self._calculate_quadratic_form(
            K_mm, K_mn
        )  # K_mn.T @ K_mm_inv @ K_mn

        cov = Q_nn + ((self.jitter + sigma_square) * jnp.eye(n))

        mvn = dist.MultivariateNormal(
            loc=jnp.zeros(n), scale_tril=jnp.linalg.cholesky(cov)
        )

        K_nn = self.kernel(x, x)
        regularizer = -1 / (2 * sigma_square) * jnp.trace(K_nn - Q_nn)

        mll = jnp.sum(mvn.log_prob(y.T))
        elbo = mll + regularizer
        return elbo

    @staticmethod
    def _solve_linear(A: Array, b: Array):
        """Solve a matrix.

        If A is symmetric and positive definite then Ax=b can be solved
        by using Cholesky decomposition.

        (1) L = cholesky(A)
        (2) L*y = b
        (3) L.T * x = y

        As L is a triangular (2) can be efficiently solved for y
        and (3) can be efficiently solved for x.

        Parameters
        ----------
        A: jax.Array
            A in term Ax=b
        b: jax.Array
            b in term Ax=b

        Returns
        -------
        float
            returns x from term Ax=b
        """
        L = jnp.linalg.cholesky(A)
        y = jsp.linalg.solve_triangular(L, b, lower=True)
        x = jsp.linalg.solve_triangular(L.T, y, lower=False)

        return x

    # pylint: disable=line-too-long
    @staticmethod
    def _calculate_quadratic_form(A: Array, x: Array):
        """Calculate the quadratic form.

        Calculates

            y = x.T * inv(A) * x

        using the Cholesky decomposition of A without actually computing inv(A)
        Note that A has to be symmetric and positive definite.

        Parameters
        ----------
        A: jax.Array
            A in term y = x.T * inv(A) * x
        x: jax.Array
            x in term y = x.T * inv(A) * x

        Returns
        -------
        float
            returns the result of x.T * inv(A) * x
        """
        L = jnp.linalg.cholesky(A)
        z = jsp.linalg.solve_triangular(L, x, lower=True)
        y = z.T @ z

        return y
