from typing import Callable

import haiku as hk
import jax
import jax.numpy as np
import numpyro.distributions as dist
from chex import assert_rank, assert_axis_dimension
from numpyro.distributions import kl_divergence

from pax.attention import uniform_attention


def latent_encoder_fn(dim):
    def _f(x):
        module = hk.Sequential([
            hk.Linear(dim * 2)
        ])
        return module(x)
    return _f


class NP(hk.Module):
    def __init__(
        self,
        deterministic_encoder: Callable,
        latent_encoder: Callable,
        latent_encoder_dim: int,
        decoder: Callable,
        attention=uniform_attention
    ):
        super(NP, self).__init__()
        self._deterministic_encoder = deterministic_encoder

        self._latent_encoder = latent_encoder
        self._latent_encoder_dim = latent_encoder_dim
        self._latent_encoder_last = latent_encoder_fn(self._latent_encoder_dim)

        self._decoder = decoder
        self._attention = attention

    def __call__(
        self,
        x_context: np.DeviceArray,
        y_context: np.DeviceArray,
        x_target: np.DeviceArray,
        **kwargs
    ):
        assert_rank([x_context, y_context, x_target], 3)
        if "y_target" in kwargs:
            assert_rank(kwargs["y_target"], 3)
            return self._elbo(x_context, y_context, x_target, **kwargs)

        _, n, _ = x_target.shape
        key = hk.next_rng_key()

        zl = self._encode_latent(x_context, y_context).sample(key)
        zd = self._encode_deterministic(x_context, y_context, x_target)
        z = self._concat_and_tile(zd, zl, n)
        mvn = self._decode(z, x_target, y_context)

        return mvn

    def _elbo(
        self,
        x_context: np.DeviceArray,
        y_context: np.DeviceArray,
        x_target: np.DeviceArray,
        y_target: np.DeviceArray
    ):
        _, n, _ = x_target.shape
        key = hk.next_rng_key()

        prior = self._encode_latent(x_context, y_context)
        posterior = self._encode_latent(x_target, y_target)

        zl = posterior.sample(key)
        zd = self._encode_deterministic(x_context, y_context, x_target)
        z = self._concat_and_tile(zd, zl, n)
        mvn = self._decode(z, x_target, y_target)

        lp__ = np.sum(mvn.log_prob(y_target), axis=1)
        kl__ = np.sum(kl_divergence(posterior, prior), axis=-1)
        elbo = np.mean(lp__ - kl__)

        return mvn, -elbo

    def _concat_and_tile(self, zd, zl, n):
        z = np.concatenate([zd, zl], axis=-1)
        assert_axis_dimension(z, 1, 1)
        z = np.tile(z, [1, n, 1])
        assert_axis_dimension(z, 1, n)
        return z

    def _encode_deterministic(
        self,
        x_context: np.DeviceArray,
        y_context: np.DeviceArray,
        x_target: np.DeviceArray
    ):
        xy_context = np.concatenate([x_context, y_context], axis=-1)
        z = self._deterministic_encoder(xy_context)
        z = self._attention(x_context, x_target, z)
        return z

    def _encode_latent(
        self,
        x_context: np.DeviceArray,
        y_context: np.DeviceArray
    ):
        xy_context = np.concatenate([x_context, y_context], axis=-1)
        z = self._latent_encoder(xy_context)
        z = np.mean(z, axis=1, keepdims=True)
        z = self._latent_encoder_last(z)
        mu, sigma = np.split(z, 2, axis=-1)
        sigma = 0.1 + 0.9 * jax.nn.sigmoid(sigma)
        return dist.Normal(loc=mu, scale=sigma)

    def _decode(
        self,
        z: np.DeviceArray,
        x_target: np.DeviceArray,
        y: np.DeviceArray,
    ):
        target = np.concatenate([z, x_target], axis=-1)
        target = self._decoder(target)
        mu, log_sigma = np.split(target, 2, axis=-1)
        sigma = 0.1 + 0.9 * jax.nn.softplus(log_sigma)

        assert_axis_dimension(mu, 0, x_target.shape[0])
        assert_axis_dimension(mu, 1, x_target.shape[1])
        assert_axis_dimension(mu, 2, y.shape[2])

        return dist.Normal(loc=mu, scale=sigma)
