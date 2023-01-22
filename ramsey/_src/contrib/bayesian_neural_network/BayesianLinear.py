from typing import Callable, Optional

import haiku as hk
import jax
from distrax import Distribution, MultivariateNormalDiag, Normal
from jax import numpy as jnp
from jax import random

from ramsey._src.contrib.bayesian_neural_network.BayesianLayer import (
    BayesianLayer,
)


# pylint: disable=too-many-instance-attributes,too-many-locals
class BayesianLinear(hk.Module, BayesianLayer):
    """
    Bayesian Linear Layer

    Bayesian Linear Layer using distributions over weights and bias.

    The KL divergences between the variational posteriors and priors
    for weigths and bias are calculated.
    The KL divergence terms can be used to obtain the ELBO as an objective
    to train a Bayesian Neural Network.

    See [1] for details.

    References
    ----------

    [1] Blundell C., Cornebise J., Kavukcuoglu K., Wierstra D.
        "Weight Uncertainty in Neural Networks".
        ICML, 2015.
    """

    def __init__(
        self,
        output_size: int,
        w_mu_init: Optional[hk.initializers.Initializer] = None,
        w_rho_init: Optional[hk.initializers.Initializer] = None,
        b_mu_init: Optional[hk.initializers.Initializer] = None,
        b_rho_init: Optional[hk.initializers.Initializer] = None,
        with_bias: bool = True,
        activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
        w_prior: Optional[Distribution] = Normal(loc=0, scale=1),
        b_prior: Optional[Distribution] = Normal(loc=0, scale=1),
        name: Optional[str] = None,
    ):
        """
        Instantiates a Bayesian Linear Layer

        Parameters
        ----------
        output_size: int
            number of layer outputs
        w_mu_init: Optional[hk.initializers.Initializer]
            init for mean of var. posterior for weigths
        w_rho_init: Optional[hk.initializers.Initializer]
            init for sigma = log(1+exp(rho)) of var. posterior for weigths
        b_mu_init: Optional[hk.initializers.Initializer]
            init for mean of var. posterior for bias
        b_rho_init: Optional[hk.initializers.Initializer]
            init for sigma = log(1+exp(rho)) of var. posterior for bias
        with_bias: bool
            control usage of bias term
        activation: Callable[[jnp.ndarray], jnp.ndarray]
            choose the activation function (use None for no activation)
        w_prior: Optional[Distribution]
            prior distriburtion for weights
        b_prior: Optional[Distribution]
            prior distribution for bias
        name: Optional[str]
            name of the layer
        """

        super().__init__(name=name)
        self._output_size = output_size
        self._with_bias = with_bias
        self._w_mu_init = w_mu_init
        self._w_rho_init = w_rho_init
        self._b_mu_init = b_mu_init or jnp.zeros
        self._b_rho_init = b_rho_init
        self._activation = activation
        self._w_prior = w_prior
        self._b_prior = b_prior

    def __call__(
        self, x: jnp.ndarray, key: random.PRNGKey, is_training: bool = False
    ):
        """
        Instantiates a sparse Gaussian process

        Parameters
        ----------
        x: jnp.ndarray
            layer inputs
        key : random.PRNGKey,
            number of inducing points
        is_training : bool
            training mode where KL divergence terms are calculated and returned
        """

        x = jnp.atleast_2d(x)
        dtype = x.dtype

        n_in = x.shape[-1]
        n_out = self._output_size

        w_mu, w_sigma = self._get_w_var_dist_params((n_in, n_out), dtype)
        W = self._reparameterize(w_mu, w_sigma, key)
        output = jnp.matmul(x, W)

        if self._with_bias:
            b_mu, b_sigma = self._get_b_var_dist_params(n_out, dtype)
            b = self._reparameterize(b_mu, b_sigma, key)
            b_ = jnp.broadcast_to(b, output.shape)
            output = output + b_

        output = self._activate(output)

        if is_training:
            kl_div = self._kl_div_w(w_mu, w_sigma, W)
            if self._with_bias:
                kl_div += self._kl_div_b(b_mu, b_sigma, b)
            return output, kl_div

        return output

    def _kl_div_b(self, mu, sigma, b):

        # variational posterior: q(b|theta)
        var_posterior = Normal(loc=mu, scale=sigma)

        # KL from posterior to prior p(b)
        kl_div = -jnp.sum(self._b_prior.log_prob(b))
        kl_div += jnp.sum(var_posterior.log_prob(b))

        return kl_div

    def _kl_div_w(self, mu, sigma, w):

        # variational posterior: q(w|theta)
        var_posterior = Normal(loc=mu, scale=sigma)

        # KL from posterior to prior p(w)
        kl_div = -jnp.sum(self._w_prior.log_prob(w))
        kl_div += jnp.sum(var_posterior.log_prob(w))

        return kl_div

    def _activate(self, x):

        output = x
        if self._activation is not None:
            output = self._activation(x)
        return output

    def _reparameterize(self, mu, sigma, key):

        e = MultivariateNormalDiag(loc=jnp.zeros(mu.shape)).sample(seed=key)
        z = mu + sigma * e
        return z

    def _get_b_var_dist_params(self, n_out, dtype):

        b_mu_var = self._get_mu_b([n_out], dtype)
        b_sigma_var = self._get_sigma_b([n_out], dtype)

        return b_mu_var, b_sigma_var

    def _get_w_var_dist_params(self, layer_dim, dtype):

        w_mu_var = self._get_mu_w(layer_dim, dtype)
        w_sigma_var = self._get_sigma_w(layer_dim, dtype)

        return w_mu_var, w_sigma_var

    def _get_mu_w(self, shape, dtype):

        n_in, _ = shape
        mu_init = self._w_mu_init

        if mu_init is None:
            stddev = 1.0 / jnp.sqrt(n_in)
            mu_init = hk.initializers.TruncatedNormal(stddev=stddev)

        mu = hk.get_parameter("w_mu", shape=shape, dtype=dtype, init=mu_init)

        return mu

    def _get_sigma_w(self, shape, dtype):

        rho_init = self._w_rho_init

        if rho_init is None:
            rho_init = hk.initializers.RandomUniform(
                self._inv_softplus(1e-2), self._inv_softplus(1e-1)
            )

        rho = hk.get_parameter("w_rho", shape=shape, dtype=dtype, init=rho_init)
        sigma = self._softplus(rho)

        return sigma

    def _get_mu_b(self, shape, dtype):

        mu = hk.get_parameter(
            "b_mu", shape=shape, dtype=dtype, init=self._b_mu_init
        )
        return mu

    def _get_sigma_b(self, shape, dtype):

        rho_init = self._b_rho_init

        if rho_init is None:
            rho_init = hk.initializers.RandomUniform(
                self._inv_softplus(1e-2), self._inv_softplus(1e-1)
            )

        rho = hk.get_parameter("b_rho", shape=shape, dtype=dtype, init=rho_init)
        sigma = self._softplus(rho)

        return sigma

    def _softplus(self, x):
        return jnp.log(1 + jnp.exp(x))

    def _inv_softplus(self, x):
        return jnp.log(jnp.exp(x) - 1)
