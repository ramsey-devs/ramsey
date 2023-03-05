from typing import Optional

import haiku as hk
from jax import numpy as jnp
from numpyro import distributions as dist


# pylint: disable=too-many-instance-attributes,too-many-locals
class BayesianLinear(hk.Linear):
    """
    Linear Bayesian layer

    A linear Bayesian layer using distributions over weights and bias.
    The KL divergences between the variational posteriors and priors
    for weigths and bias are calculated. The KL divergence terms can be
    used to obtain the ELBO as an objective to train a Bayesian neural network.

    References
    ----------

    [1] Blundell C., Cornebise J., Kavukcuoglu K., Wierstra D.
        "Weight Uncertainty in Neural Networks". ICML, 2015.
    """

    def __init__(
        self,
        output_size: int,
        with_bias: bool = True,
        w_prior: Optional[dist.Distribution] = dist.Normal(loc=0.0, scale=1.0),
        b_prior: Optional[dist.Distribution] = dist.Normal(loc=0.0, scale=1.0),
        name: Optional[str] = None,
        **kwargs,
    ):
        """
        Instantiates a linear Bayesian layer

        Parameters
        ----------
        output_size: int
            number of layer outputs
        with_bias: bool
            control usage of bias term
        w_prior: Optional[Distribution]
            prior distriburtion for weights
        b_prior: Optional[Distribution]
            prior distribution for bias
        name: Optional[str]
            name of the layer
        kwargs: keyword arguments
            you can supply the initializers for the parameters of the priors using the
            keyword arguments. For instance, if your prior on the weights
            is a dist.Normal(loc, scale) then you can supply
            hk.initializers.Initializer objects with names w_loc_init and
            w_scale_init as keyword arguments. Likewise you can supply
            initializers called b_loc_init and b_scale_init for the prior on the
            bias. If your prior on the weights is a dist.Uniform(low, high)
            you will need to supply initializers called w_low_init and
            w_high_init
        """

        dist.Uniform()
        super().__init__(name=name)
        self._output_size = output_size
        self._with_bias = with_bias
        self._w_prior = w_prior
        self._b_prior = b_prior
        self._kwargs = kwargs

    def __call__(self, x: jnp.ndarray, is_training: bool = False):
        """
        Instantiates a sparse Gaussian process

        Parameters
        ----------
        x: jnp.ndarray
            layer inputs
        is_training : bool
            training mode where KL divergence terms are calculated and returned
        """

        dtype = x.dtype
        n_in = x.shape[-1]

        w_mu, w_sigma = self._get_w_var_dist_params(
            (n_in, self._output_size), dtype
        )
        W = self._reparameterize(w_mu, w_sigma)
        output = jnp.matmul(x, W)

        if self._with_bias:
            b_mu, b_sigma = self._get_b_var_dist_params(
                self._output_size, dtype
            )
            b = self._reparameterize(b_mu, b_sigma)
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
        var_posterior = dist.Normal(loc=mu, scale=sigma)

        # KL from posterior to prior p(b)
        kl_div = -jnp.sum(self._b_prior.log_prob(b))
        kl_div += jnp.sum(var_posterior.log_prob(b))

        return kl_div

    def _kl_div_w(self, mu, sigma, w):

        # variational posterior: q(w|theta)
        var_posterior = dist.Normal(loc=mu, scale=sigma)

        # KL from posterior to prior p(w)
        kl_div = -jnp.sum(self._w_prior.log_prob(w))
        kl_div += jnp.sum(var_posterior.log_prob(w))

        return kl_div

    @staticmethod
    def _reparameterize(mu, sigma):
        e = dist.MultivariateNormal(loc=jnp.zeros(mu.shape)).sample(
            hk.next_rng_key()
        )
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
