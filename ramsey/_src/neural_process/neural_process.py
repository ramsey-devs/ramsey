from typing import Optional, Tuple

import jax
from flax import linen as nn
from jax import Array, numpy as jnp

import numpyro.distributions as dist
from chex import assert_axis_dimension, assert_rank
from numpyro.distributions import kl_divergence

from ramsey._src.family import Family, Gaussian

__all__ = ["NP"]


# pylint: disable=too-many-instance-attributes,duplicate-code
class NP(nn.Module):
    """
    A neural process.

    Implements the core structure of a neural process [1], i.e.,
    an optional deterministic encoder, a latent encoder, and a decoder.

    Attributes
    ----------
    decoder: flax.linen.Module
        the decoder can be any network, but is typically an MLP. Note
        that the _last_ layer of the decoder needs to
        have twice the number of nodes as the data you try to model
    latent_encoders: Tuple[flax.linen.Module, flax.linen.Module]
        a tuple of two `flax.linen.Module`s. The latent encoder can be any network,
        but is typically an MLP. The first element of the tuple is a neural
        network used before the aggregation step, while the second element
        of the tuple encodes is a neural network used to
        compute mean(s) and standard deviation(s) of the latent Gaussian.
    deterministic_encoder: Optional[flax.linen.Module]
        the deterministic encoder can be any network, but is typically an
        MLP
    family: Family
        distributional family of the response variable

    References
    ----------
    .. [1] Garnelo, Marta, et al. "Neural processes".
       CoRR. 2018.
    """

    decoder: nn.Module
    latent_encoder: Tuple[nn.Module, nn.Module]
    deterministic_encoder: Optional[nn.Module] = None
    family: Family = Gaussian()

    def setup(self):
        self._deterministic_encoder = self.deterministic_encoder
        self._decoder = self.decoder
        self._family = self.family
        [self._latent_encoder, self._latent_variable_encoder] = (
            self.latent_encoder[0],
            self.latent_encoder[1]
        )

    @nn.compact
    def __call__(
        self,
        inputs_context: Array,
        outputs_context: Array,
        inputs_target: Array,
        **kwargs,
    ):
        """
        Transform the inputs through the neural process.

        Parameters
        ----------
        inputs_context: jax.Array
            Input data of dimension (*batch_dims, spatial_dims..., feature_dims)
        outputs_context: jax.Array
            Input data of dimension (*batch_dims, spatial_dims..., response_dims)
        inputs_target: jax.Array
            Input data of dimension (*batch_dims, spatial_dims..., feature_dims)
        **kwargs: kwargs
            Keyword arguments can include:
            - outputs_target: jax.Array. If an argument called 'outputs_target'
            is provided, computes the loss (negative ELBO) together with a
            predictive posterior distribution

        Returns
        -------
        Union[numpyro.distribution, Tuple[numpyro.distribution, float]]
            If 'outputs_target' is provided as keyword argument, returns a tuple
            of the predictive distribution and the negative ELBO which can be
            used as loss for optimization.
            If 'outputs_target' is not provided, returns the predictive
            distribution only.
        """

        assert_rank([inputs_context, outputs_context, inputs_target], 3)
        if "outputs_target" in kwargs:
            assert_rank(kwargs["outputs_target"], 3)
            return self._negative_elbo(
                inputs_context, outputs_context, inputs_target, **kwargs
            )

        _, num_observations, _ = inputs_target.shape

        rng = self.make_rng('sample')
        z_latent = self._encode_latent(inputs_context, outputs_context).sample(rng)
        z_deterministic = self._encode_deterministic(
            inputs_context, outputs_context, inputs_target
        )
        representation = self._concat_and_tile(
            z_deterministic, z_latent, num_observations
        )
        pred_fn = self._decode(representation, inputs_target, outputs_context)

        return pred_fn

    def _negative_elbo(  # pylint: disable=too-many-locals
        self,
        inputs_context: Array,
        outputs_context: Array,
        inputs_target: Array,
        outputs_target: Array,
    ):
        _, num_observations, _ = inputs_target.shape

        prior = self._encode_latent(inputs_context, outputs_context)
        posterior = self._encode_latent(inputs_target, outputs_target)

        rng = self.make_rng('sample')
        z_latent = posterior.sample(rng)
        z_deterministic = self._encode_deterministic(
            inputs_context, outputs_context, inputs_target
        )
        representation = self._concat_and_tile(
            z_deterministic, z_latent, num_observations
        )
        pred_fn = self._decode(representation, inputs_target, outputs_target)

        loglik = jnp.sum(pred_fn.log_prob(outputs_target), axis=1)
        kl = jnp.sum(kl_divergence(posterior, prior), axis=-1)
        elbo = jnp.mean(loglik - kl)

        return pred_fn, -elbo

    @staticmethod
    # pylint: disable=duplicate-code
    def _concat_and_tile(z_deterministic, z_latent, num_observations):
        if z_deterministic is None:
            representation = z_latent
        else:
            representation = jnp.concatenate(
                [z_deterministic, z_latent], axis=-1
            )
        assert_axis_dimension(representation, 1, 1)
        representation = jnp.tile(representation, [1, num_observations, 1])
        assert_axis_dimension(representation, 1, num_observations)
        return representation

    def _encode_deterministic(
        self,
        inputs_context: Array,
        outputs_context: Array,
        inputs_target: Array,  # pylint: disable=unused-argument
    ):
        if self._deterministic_encoder is None:
            return None
        xoutputs_context = jnp.concatenate([inputs_context, outputs_context], axis=-1)
        z_deterministic = self._deterministic_encoder(xoutputs_context)
        z_deterministic = jnp.mean(z_deterministic, axis=1, keepdims=True)
        return z_deterministic

    def _encode_latent(
            self,
            inputs_context: Array,
            outputs_context: Array
    ):
        xoutputs_context = jnp.concatenate([inputs_context, outputs_context], axis=-1)
        z_latent = self._latent_encoder(xoutputs_context)
        return self._encode_latent_gaussian(z_latent)

    # pylint: disable=duplicate-code
    def _encode_latent_gaussian(self, z_latent):
        z_latent = jnp.mean(z_latent, axis=1, keepdims=True)
        z_latent = self._latent_variable_encoder(z_latent)
        mean, sigma = jnp.split(z_latent, 2, axis=-1)
        sigma = 0.1 + 0.9 * jax.nn.sigmoid(sigma)
        return dist.Normal(loc=mean, scale=sigma)

    def _decode(
        self,
        representation: Array,
        inputs_target: Array,
        y: Array
    ):
        target = jnp.concatenate([representation, inputs_target], axis=-1)
        target = self._decoder(target)
        family = self._family(target)
        self._check_posterior_predictive_axis(family, inputs_target, y)
        return family

    @staticmethod
    def _check_posterior_predictive_axis(
        family: dist.Distribution,
        inputs_target: Array,
        y: Array,  # pylint: disable=invalid-name
    ):
        assert_axis_dimension(family.mean, 0, inputs_target.shape[0])
        assert_axis_dimension(family.mean, 1, inputs_target.shape[1])
        assert_axis_dimension(family.mean, 2, y.shape[2])
