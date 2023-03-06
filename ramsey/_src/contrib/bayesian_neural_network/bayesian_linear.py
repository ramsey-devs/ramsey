from typing import Optional, Tuple

import haiku as hk
from jax import numpy as jnp
from numpyro import distributions as dist
from numpyro.distributions import constraints, kl_divergence


# pylint: disable=too-many-instance-attributes,too-many-locals
class BayesianLinear(hk.Module):
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
            prior distribution for weights
        b_prior: Optional[Distribution]
            prior distribution for bias
        name: Optional[str]
            name of the layer
        kwargs: keyword arguments
            you can supply the initializers for the parameters of the priors
            using the keyword arguments. For instance, if your prior on the
            weights is a dist.Normal(loc, scale) then you can supply
            hk.initializers.Initializer objects with names w_loc_init and
            w_scale_init as keyword arguments. Likewise you can supply
            initializers called b_loc_init and b_scale_init for the prior on the
            bias. If your prior on the weights is a dist.Uniform(low, high)
            you will need to supply initializers called w_low_init and
            w_high_init
        """

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
        w, w_params = self._get_weights((n_in, self._output_size), dtype)
        output = jnp.dot(x, w)
        if self._with_bias:
            b, b_params = self._get_bias(self._output_size, dtype)
            b = jnp.broadcast_to(b, output.shape)
            output = output + b
        if is_training:
            kl_div = self._kl(self._w_prior, w_params)
            if self._with_bias:
                kl_div += self._kl(self._b_prior, b_params)
            return output, kl_div
        return output

    def _get_weights(self, layer_dim, dtype):
        arg_constraints = self._w_prior.arg_constraints
        params = {
            param_name: self._init_param(
                "w", param_name, constraint, layer_dim, dtype
            )
            for param_name, constraint in arg_constraints.items()
        }
        samples = self._w_prior.__class__(**params).sample(hk.next_rng_key())
        return samples, params

    def _get_bias(self, layer_dim, dtype):
        arg_constraints = self._b_prior.arg_constraints
        params = {
            param_name: self._init_param(
                "b", param_name, constraint, layer_dim, dtype
            )
            for param_name, constraint in arg_constraints.items()
        }
        samples = self._b_prior.__class__(**params).sample(hk.next_rng_key())
        return samples, params

    def _init_param(self, weight_name, param_name, constraint, shape, dtype):
        init_name = f"{weight_name}_{param_name}_init"
        if init_name in self._kwargs:
            init = self._kwargs[init_name]
        else:
            init = hk.initializers.TruncatedNormal(stddev=1)
        shape = (shape,) if not isinstance(shape, Tuple) else shape
        params = hk.get_parameter(
            f"{weight_name}_{param_name}", shape=shape, dtype=dtype, init=init
        )
        params = jnp.where(
            constraints.positive == constraint, jnp.exp(params), params
        )
        return params

    @staticmethod
    def _kl(prior, params):
        var_posterior = dist.Normal(**params)
        kl_divergence(var_posterior, prior)
        return jnp.sum(kl_divergence(var_posterior, prior))
