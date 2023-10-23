from typing import Optional, Tuple

import flax
import jax
import numpyro.distributions as dist
from chex import assert_axis_dimension, assert_rank
from flax import linen as nn
from jax import Array
from jax import numpy as jnp
from numpyro.distributions import kl_divergence

from ramsey._src.family import Family, Gaussian

__all__ = ["NP"]


# pylint: disable=too-many-instance-attributes,duplicate-code,not-callable
class NP(nn.Module):
    """A neural process.

    Implements the core structure of a neural process [1], i.e.,
    an optional deterministic encoder, a latent encoder, and a decoder.

    Attributes
    ----------
    decoder: flax.linen.Module
        the decoder can be any network, but is typically an MLP. Note
        that the _last_ layer of the decoder needs to
        have twice the number of nodes as the data you try to model
    latent_encoder: Optional[Tuple[flax.linen.Module, flax.linen.Module]]
        a tuple of two `flax.linen.Module`s. The latent encoder can be
        any network, but is typically an MLP. The first element of the tuple
        is a neural network used before the aggregation step, while the second
        element of the tuple encodes is a neural network used to
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
    latent_encoder: Optional[Tuple[flax.linen.Module, flax.linen.Module]] = None
    deterministic_encoder: Optional[flax.linen.Module] = None
    family: Family = Gaussian()

    def setup(self):
        """Construct the networks of the class."""
        if self.latent_encoder is None and self.deterministic_encoder is None:
            raise ValueError(
                "either latent or deterministic encoder needs to be set"
            )

        self._decoder = self.decoder
        if self.latent_encoder is not None:
            [self._latent_encoder, self._latent_variable_encoder] = (
                self.latent_encoder[0],
                self.latent_encoder[1],
            )
        if self.deterministic_encoder is not None:
            self._deterministic_encoder = self.deterministic_encoder
        self._family = self.family

    @nn.compact
    def __call__(
        self,
        x_context: Array,
        y_context: Array,
        x_target: Array,
        **kwargs,
    ):
        """Transform the inputs through the neural process.

        Parameters
        ----------
        x_context: jax.Array
            Input data of dimension (*batch_dims, spatial_dims..., feature_dims)
        y_context: jax.Array
            Input data of dimension
            (*batch_dims, spatial_dims..., response_dims)
        x_target: jax.Array
            Input data of dimension (*batch_dims, spatial_dims..., feature_dims)
        **kwargs: kwargs
            Keyword arguments can include:
            - y_target: jax.Array. If an argument called 'y_target'
            is provided, computes the loss (negative ELBO) together with a
            predictive posterior distribution

        Returns
        -------
        Union[numpyro.distribution, Tuple[numpyro.distribution, float]]
            If 'y_target' is provided as keyword argument, returns a tuple
            of the predictive distribution and the negative ELBO which can be
            used as loss for optimization.
            If 'y_target' is not provided, returns the predictive
            distribution only.
        """
        assert_rank([x_context, y_context, x_target], 3)
        if "y_target" in kwargs:
            assert_rank(kwargs["y_target"], 3)
            return self._negative_elbo(x_context, y_context, x_target, **kwargs)

        _, num_observations, _ = x_target.shape

        if self.latent_encoder is not None:
            rng = self.make_rng("sample")
            z_latent = self._encode_latent(x_context, y_context).sample(rng)
        else:
            z_latent = None

        z_deterministic = self._encode_deterministic(
            x_context, y_context, x_target
        )
        representation = self._concat_and_tile(
            z_deterministic, z_latent, num_observations
        )
        pred_fn = self._decode(representation, x_target, y_context)

        return pred_fn

    def _negative_elbo(  # pylint: disable=too-many-locals
        self,
        x_context: Array,
        y_context: Array,
        x_target: Array,
        y_target: Array,
    ):
        _, num_observations, _ = x_target.shape

        if self.latent_encoder is not None:
            rng = self.make_rng("sample")
            prior = self._encode_latent(x_context, y_context)
            posterior = self._encode_latent(x_target, y_target)
            z_latent = posterior.sample(rng)
            kl = jnp.sum(kl_divergence(posterior, prior), axis=-1)
        else:
            z_latent = None
            kl = 0

        z_deterministic = self._encode_deterministic(
            x_context, y_context, x_target
        )
        representation = self._concat_and_tile(
            z_deterministic, z_latent, num_observations
        )
        pred_fn = self._decode(representation, x_target, y_target)
        loglik = jnp.sum(pred_fn.log_prob(y_target), axis=1)
        elbo = jnp.mean(loglik - kl)

        return pred_fn, -elbo

    @staticmethod
    # pylint: disable=duplicate-code
    def _concat_and_tile(z_deterministic, z_latent, num_observations):
        if z_deterministic is None:
            representation = z_latent
        elif z_latent is None:
            representation = z_deterministic
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
        x_context: Array,
        y_context: Array,
        x_target: Array,  # pylint: disable=unused-argument
    ):
        if self.deterministic_encoder is None:
            return None
        xy_context = jnp.concatenate([x_context, y_context], axis=-1)
        z_deterministic = self._deterministic_encoder(xy_context)
        z_deterministic = jnp.mean(z_deterministic, axis=1, keepdims=True)
        return z_deterministic

    def _encode_latent(self, x_context: Array, y_context: Array):
        xy_context = jnp.concatenate([x_context, y_context], axis=-1)
        z_latent = self._latent_encoder(xy_context)
        return self._encode_latent_gaussian(z_latent)

    # pylint: disable=duplicate-code
    def _encode_latent_gaussian(self, z_latent):
        z_latent = jnp.mean(z_latent, axis=1, keepdims=True)
        z_latent = self._latent_variable_encoder(z_latent)
        mean, sigma = jnp.split(z_latent, 2, axis=-1)
        sigma = 0.1 + 0.9 * jax.nn.sigmoid(sigma)
        return dist.Normal(loc=mean, scale=sigma)

    def _decode(self, representation: Array, x_target: Array, y: Array):
        target = jnp.concatenate([representation, x_target], axis=-1)
        target = self._decoder(target)
        family = self._family(target)
        self._check_posterior_predictive_axis(family, x_target, y)
        return family

    @staticmethod
    def _check_posterior_predictive_axis(
        family: dist.Distribution,
        x_target: Array,
        y: Array,  # pylint: disable=invalid-name
    ):
        assert_axis_dimension(family.mean, 0, x_target.shape[0])
        assert_axis_dimension(family.mean, 1, x_target.shape[1])
        assert_axis_dimension(family.mean, 2, y.shape[2])
