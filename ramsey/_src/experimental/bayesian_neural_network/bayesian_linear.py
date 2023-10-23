from typing import Optional, Tuple

from flax import linen as nn
from flax.linen import initializers
from jax import Array
from jax import numpy as jnp
from numpyro import distributions as dist
from numpyro.distributions import constraints, kl_divergence


# pylint: disable=too-many-instance-attributes,too-many-locals
class BayesianLinear(nn.Module):
    """Linear Bayesian layer.

    A linear Bayesian layer using distributions over weights and bias.
    The KL divergences between the variational posteriors and priors
    for weigths and bias are calculated. The KL divergence terms can be
    used to obtain the ELBO as an objective to train a Bayesian neural network.

    Attributes
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

    References
    ----------
    .. [1] Blundell C., Cornebise J., Kavukcuoglu K., Wierstra D.
        "Weight Uncertainty in Neural Networks". ICML, 2015.
    """

    output_size: int
    use_bias: bool = True
    mc_sample_size: int = 10
    w_prior: Optional[dist.Distribution] = dist.Normal(loc=0.0, scale=1.0)
    b_prior: Optional[dist.Distribution] = dist.Normal(loc=0.0, scale=1.0)
    name: Optional[str] = None

    def setup(self):
        """Construct a linear Bayesian layer."""
        self._output_size = self.output_size
        self._with_bias = self.use_bias
        self._w_prior = self.w_prior
        self._b_prior = self.b_prior

    @nn.compact
    def __call__(self, x: Array, is_training: bool = False):
        """Call a sparse Gaussian process.

        Parameters
        ----------
        inputs: jax.Array
            layer inputs
        is_training : bool
            training mode where KL divergence terms are calculated and returned
        """
        dtype = x.dtype
        n_in = x.shape[-1]
        outputs = x
        w, w_params = self._get_weights((n_in, self._output_size), dtype)
        outputs = jnp.einsum("...bj,...sjk->sbk", outputs, w)
        if self._with_bias:
            b, b_params = self._get_bias((1, self._output_size), dtype)
            b = jnp.broadcast_to(b, outputs.shape)
            outputs = outputs + b
        outputs = jnp.mean(outputs, axis=0)

        if is_training:
            kl_div = self._kl(self._w_prior, w_params)
            if self._with_bias:
                kl_div += self._kl(self._b_prior, b_params)
            return outputs, kl_div
        return outputs

    def _get_weights(self, layer_dim, dtype):
        arg_constraints = self._w_prior.arg_constraints
        params = {
            param_name: self._init_param(
                "w", param_name, constraint, layer_dim, dtype
            )
            for param_name, constraint in arg_constraints.items()
        }
        samples = self._w_prior.__class__(**params).sample(
            self.make_rng("sample"), (self.mc_sample_size,)
        )
        return samples, params

    def _get_bias(self, layer_dim, dtype):
        arg_constraints = self._b_prior.arg_constraints
        params = {
            param_name: self._init_param(
                "b", param_name, constraint, layer_dim, dtype
            )
            for param_name, constraint in arg_constraints.items()
        }
        samples = self._b_prior.__class__(**params).sample(
            self.make_rng("sample"), (self.mc_sample_size,)
        )
        return samples, params

    def _init_param(self, weight_name, param_name, constraint, shape, dtype):
        init = initializers.xavier_normal()

        shape = (shape,) if not isinstance(shape, Tuple) else shape
        params = self.param(f"{weight_name}_{param_name}", init, shape, dtype)

        params = jnp.where(
            constraints.positive == constraint, jnp.exp(params), params
        )
        return params

    @staticmethod
    def _kl(prior, params):
        var_posterior = dist.Normal(**params)
        kl = kl_divergence(var_posterior, prior)
        return jnp.sum(kl)
