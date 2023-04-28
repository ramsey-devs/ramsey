from typing import Optional, Tuple, Union

import haiku as hk
import jax
import numpyro.distributions as dist
from chex import assert_axis_dimension, assert_rank
from jax import numpy as jnp
from numpyro.distributions import kl_divergence

from ramsey._src.attention.attention import Attention
from ramsey.family import Family, Gaussian

__all__ = ["GNP"]


# pylint: disable=too-many-instance-attributes,duplicate-code
class GNP(hk.Module):
    def __init__(
        self,
        decoder: Union[hk.Module, hk.DeepRNN],
        latent_encoder: Optional[
            Union[
                Tuple[hk.Module, hk.Module],
                Tuple[hk.DeepRNN, hk.Module],
                Tuple[hk.Module, Attention, hk.Module],
            ],
        ] = None,
        deterministic_encoder: Optional[
            Union[
                hk.Module,
                Tuple[hk.Module, Attention],
                Tuple[hk.Module, Attention, Attention],
            ]
        ] = None,
        family: Family = Gaussian(),
    ):
        super().__init__()
        self._decoder = decoder
        if latent_encoder is not None:
            if len(latent_encoder) == 2:
                [self._latent_encoder, self._latent_variable_encoder] = (
                    latent_encoder[0],
                    latent_encoder[1],
                )
            elif len(latent_encoder) == 3:
                [
                    self._latent_encoder,
                    self._latent_self_attention,
                    self._latent_variable_encoder,
                ] = (latent_encoder[0], latent_encoder[1], latent_encoder[2])
            else:
                raise ValueError(
                    "latent envoder should be None or a tuple of length 2 or 3"
                )
        if deterministic_encoder is not None:
            if not isinstance(deterministic_encoder, Tuple):
                self._deterministic_encoder = deterministic_encoder
            elif len(deterministic_encoder) == 2:
                self._deterministic_encoder = deterministic_encoder[0]
                self._deterministic_cross_attention = deterministic_encoder[1]
            elif len(deterministic_encoder) == 3:
                self._deterministic_encoder = deterministic_encoder[0]
                self._deterministic_self_attention = deterministic_encoder[1]
                self._deterministic_cross_attention = deterministic_encoder[2]
        self._family = family

    def __call__(
        self,
        x_context: jnp.ndarray,
        y_context: jnp.ndarray,
        x_target: jnp.ndarray,
        **kwargs,
    ):
        assert_rank([x_context, y_context, x_target], 3)
        if "y_target" in kwargs:
            assert_rank(kwargs["y_target"], 3)
            return self._negative_elbo(x_context, y_context, x_target, **kwargs)

        _, num_observations, _ = x_target.shape

        z_latent = self._encode_latent(x_context, y_context).sample(
            hk.next_rng_key()
        )
        z_deterministic = self._encode_deterministic(
            x_context, y_context, x_target
        )
        representation = self._concat_and_tile(
            z_deterministic, z_latent, num_observations
        )
        mvn = self._decode(representation, x_target, y_context)

        return mvn

    def _negative_elbo(  # pylint: disable=too-many-locals
        self,
        x_context: jnp.ndarray,
        y_context: jnp.ndarray,
        x_target: jnp.ndarray,
        y_target: jnp.ndarray,
    ):
        _, num_observations, _ = x_target.shape
        key = hk.next_rng_key()

        prior = self._encode_latent(x_context, y_context)
        posterior = self._encode_latent(x_target, y_target)

        z_latent = posterior.sample(key)
        z_deterministic = self._encode_deterministic(
            x_context, y_context, x_target
        )
        representation = self._concat_and_tile(
            z_deterministic, z_latent, num_observations
        )
        mvn = self._decode(representation, x_target, y_target)

        lpp__ = jnp.sum(mvn.log_prob(y_target), axis=-1, keepdims=True)
        kl__ = jnp.sum(kl_divergence(posterior, prior), axis=-1)
        elbo = jnp.sum(lpp__, axis=1) - kl__

        return mvn, -elbo, lpp__, kl__

    def _encode_latent(self, x_context: jnp.ndarray, y_context: jnp.ndarray):
        xy_context = jnp.concatenate([x_context, y_context], axis=-1)
        if self._latent_encoder is not None:
            z_latent = self._apply_possibly_recurrent_net(
                xy_context, self._latent_encoder
            )
            if self._latent_self_attention is not None:
                z_latent = self._latent_self_attention(
                    z_latent, z_latent, z_latent
                )
            return self._encode_latent_gaussian(z_latent)
        return None

    def _apply_possibly_recurrent_net(self, x, fn):
        if isinstance(fn, hk.DeepRNN):
            return self._unroll_recurrent(x, fn)
        else:
            z_latent = fn(x)
        return z_latent

    @staticmethod
    def _unroll_recurrent(x, model):
        n_batch, _, _ = x.shape
        target, _ = hk.dynamic_unroll(
            model,
            x,
            model.initial_state(n_batch),
            time_major=False,
        )
        return target

    # pylint: disable=duplicate-code
    def _encode_latent_gaussian(self, z_latent):
        z_latent = jnp.mean(z_latent, axis=1, keepdims=True)
        z_latent = self._latent_variable_encoder(z_latent)
        mean, sigma = jnp.split(z_latent, 2, axis=-1)
        sigma = 0.1 + 0.9 * jax.nn.sigmoid(sigma)
        return dist.Normal(loc=mean, scale=sigma)

    def _encode_deterministic(
        self,
        x_context: jnp.ndarray,
        y_context: jnp.ndarray,
        x_target: jnp.ndarray,
    ):
        xy_context = jnp.concatenate([x_context, y_context], axis=-1)

        if self._deterministic_encoder is not None:
            z_deterministic = self._apply_possibly_recurrent_net(
                xy_context, self._deterministic_encoder
            )
        else:
            return None
        if self._deterministic_cross_attention is not None:
            z_deterministic = self._deterministic_cross_attention(
                x_context, z_deterministic, x_target
            )
        else:
            z_deterministic = jnp.mean(z_deterministic, axis=1, keepdims=True)
        return z_deterministic

    @staticmethod
    # pylint: disable=duplicate-code
    def _concat_and_tile(z_deterministic, z_latent, num_observations):
        if z_deterministic is None and z_latent is None:
            return None
        elif z_deterministic is None:
            representation = z_latent
        elif (
            z_latent.shape[1] == 1
            and z_deterministic.shape[1] == num_observations
        ):
            z_latent = jnp.tile(z_latent, [1, num_observations, 1])
            representation = jnp.concatenate(
                [z_deterministic, z_latent], axis=-1
            )
        else:
            representation = jnp.concatenate(
                [z_deterministic, z_latent], axis=-1
            )
            representation = jnp.tile(representation, [1, num_observations, 1])
        assert_axis_dimension(representation, 1, num_observations)
        return representation

    def _decode(
        self,
        representation: jnp.ndarray,
        x_target: np.ndarray,
        y: np.ndarray,  # pylint: disable=invalid-name
    ):
        target = jnp.concatenate([representation, x_target], axis=-1)
        target = self._apply_possibly_recurrent_net(target, self._decoder)
        family = self._family(target)
        self._check_posterior_predictive_axis(family, x_target, y)
        return family

    @staticmethod
    def _check_posterior_predictive_axis(
        family: dist.Distribution,
        x_target: np.ndarray,
        y: np.ndarray,  # pylint: disable=invalid-name
    ):
        assert_axis_dimension(family.mean, 0, x_target.shape[0])
        assert_axis_dimension(family.mean, 1, x_target.shape[1])
        assert_axis_dimension(family.mean, 2, y.shape[2])
