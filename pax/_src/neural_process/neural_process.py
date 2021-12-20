from typing import Callable, Tuple, Union

import haiku as hk
import jax
import jax.numpy as np
import numpyro.distributions as dist
from chex import assert_axis_dimension, assert_rank
from numpyro.distributions import kl_divergence

from pax._src.family import Family, Gaussian
from pax._src.neural_process.attention import uniform_attention

__all__ = ["NP"]


def _get_attention(attention: str):
    attentions = ["uniform"]
    if attention == "uniform":
        return uniform_attention
    raise ValueError(f"attention type should be in {'/'.join(attentions)}")


class NP(hk.Module):
    """
    A neural process

    Implements the core structure of a neural process, i.e., two encoders
    and a decoder as haiku module.
    """

    def __init__(
        self,
        decoder: Union[Callable, hk.Module],
        latent_encoder: Tuple[
            Union[Callable, hk.Module], Union[Callable, hk.Module]
        ],
        deterministic_encoder: Union[Callable, hk.Module, None] = None,
        family: Family = Gaussian(),
        attention="uniform",
    ):
        """
        Constructor

        Instantiates a neural process. Needs to be called directly within
        `hk.transform` with the respective arguments.

        Parameters
        ----------
        decoder: Union[Callable, hk.Module]
            either a function that wraps an `hk.Module` and calls it or a
            `hk.Module`. The decoder can be any network, but is
            typically an MLP. Note that the _last_ layer of the decoder needs to
            have twice the number of nodes as the data you try to model!
            That means if your response is univariate
        latent_encoder:  tuple[Union[Callable, hk.Module],
                               Union[Callable, hk.Module]]
            a tuple of either functions that wrap `hk.Module`s and calls them or
            two `hk.Module`s. The latent encoder can be any network, but is
            typically an MLP. The first element of the tuple is a neural network
            used before the aggregation step, while the second element of
            the tuple encodes is a neural network used to
            compute mean(s) and standard deviation(s) of the latent Gaussian.
        deterministic_encoder: Union[Callable, hk.Module, None]
            either a function that wraps an `hk.Module` and calls it or a
            `hk.Module`. The deterministic encoder can be any network, but is
            typically an MLP
        family: Family
            distributional family of the response variable
        attention: str
            attention type to apply. Can be either of 'uniform'.
        """

        super().__init__()
        self._deterministic_encoder = deterministic_encoder
        if len(latent_encoder) != 2:
            raise ValueError(
                "'latent_encoder' needs to be a tuple containing "
                "two haiku modules"
            )
        self._latent_encoder = latent_encoder[0]
        self._latent_encoder_last = latent_encoder[1]
        self._decoder = decoder
        self._family = family
        self._attention = _get_attention(attention)

    def __call__(
        self,
        x_context: np.ndarray,
        y_context: np.ndarray,
        x_target: np.ndarray,
        **kwargs,
    ):
        assert_rank([x_context, y_context, x_target], 3)
        if "y_target" in kwargs:
            assert_rank(kwargs["y_target"], 3)
            return self._elbo(x_context, y_context, x_target, **kwargs)

        _, num_observations, _ = x_target.shape
        key = hk.next_rng_key()

        z_latent = self._encode_latent(x_context, y_context).sample(key)
        z_deterministic = self._encode_deterministic(
            x_context, y_context, x_target
        )
        representation = self._concat_and_tile(
            z_deterministic, z_latent, num_observations
        )
        mvn = self._decode(representation, x_target, y_context)

        return mvn

    def _elbo(  # pylint: disable=too-many-locals
        self,
        x_context: np.ndarray,
        y_context: np.ndarray,
        x_target: np.ndarray,
        y_target: np.ndarray,
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

        lp__ = np.sum(mvn.log_prob(y_target), axis=1)
        kl__ = np.sum(kl_divergence(prior, posterior), axis=-1)
        elbo = np.mean(lp__ - kl__)

        return mvn, -elbo

    @staticmethod
    def _concat_and_tile(z_deterministic, z_latent, num_observations):
        if z_deterministic is None:
            representation = z_latent
        else:
            representation = np.concatenate(
                [z_deterministic, z_latent], axis=-1
            )
        assert_axis_dimension(representation, 1, 1)
        representation = np.tile(representation, [1, num_observations, 1])
        assert_axis_dimension(representation, 1, num_observations)
        return representation

    def _encode_deterministic(
        self,
        x_context: np.ndarray,
        y_context: np.ndarray,
        x_target: np.ndarray,
    ):
        if self._deterministic_encoder is None:
            return None
        xy_context = np.concatenate([x_context, y_context], axis=-1)
        z_deterministic = self._deterministic_encoder(xy_context)
        z_deterministic = self._attention(x_context, x_target, z_deterministic)
        return z_deterministic

    def _encode_latent(self, x_context: np.ndarray, y_context: np.ndarray):
        xy_context = np.concatenate([x_context, y_context], axis=-1)
        z_latent = self._latent_encoder(xy_context)
        z_latent = np.mean(z_latent, axis=1, keepdims=True)
        z_latent = self._latent_encoder_last(z_latent)

        mean, sigma = np.split(z_latent, 2, axis=-1)
        sigma = 0.1 + 0.9 * jax.nn.sigmoid(sigma)
        return dist.Normal(loc=mean, scale=sigma)

    def _decode(
        self,
        representation: np.ndarray,
        x_target: np.ndarray,
        y: np.ndarray,  # pylint: disable=invalid-name
    ):
        target = np.concatenate([representation, x_target], axis=-1)
        target = self._decoder(target)
        return self._family(target, x_target, y)
