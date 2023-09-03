from typing import Tuple

from chex import assert_axis_dimension, assert_rank
from flax import linen as nn
from jax import Array
from jax import numpy as jnp

from ramsey import ANP, Attention
from ramsey.family import Family, Gaussian

__all__ = ["RANP"]


# pylint: disable=too-many-instance-attributes,duplicate-code
class RANP(ANP):
    """
    A recurrent attentive neural process

    Implements the core structure of a recurrent attentive neural process
    cross-attention module.
    """

    decoder: nn.Module
    latent_encoder: Tuple[nn.Module, nn.Module]
    deterministic_encoder: Tuple[nn.Module, Attention]
    family: Family = Gaussian()

    def setup(self):
        """
        Instantiates a recurrent attentive neural process

        Parameters
        ----------
        decoder: nn.Sequential
            the decoder can be any network, but is typically an MLP. Note
            that the _last_ layer of the decoder needs to
            have twice the number of nodes as the data you try to model
        latent_encoder: Tuple[nn.Module, nn.Module]
            a tuple of two `nn.Module`s. The latent encoder can be any network,
            but is typically an MLP. The first element of the tuple is a neural
            network used before the aggregation step, while the second element
            of the tuple encodes is a neural network used to
            compute mean(s) and standard deviation(s) of the latent Gaussian.
        deterministic_encoder: Tuple[nn.Module, Attention]
            a tuple of a `nn.Module` and an Attention object. The deterministic
            encoder can be any network, but is typically an MLP
        family: Family
            distributional family of the response variable
        """

        self._decoder = self.decoder
        (self._latent_encoder, self._latent_variable_encoder) = (
            self.latent_encoder[0],
            self.latent_encoder[1],
        )
        self._deterministic_encoder = self.deterministic_encoder[0]
        self._deterministic_cross_attention = self.deterministic_encoder[1]
        self._family = self.family

    def _decode(
        self,
        representation: Array,
        x_target: Array,
        y: Array,  # pylint: disable=invalid-name
    ):
        target = jnp.concatenate([representation, x_target], axis=-1)
        assert_rank(target, 3)
        assert_axis_dimension(target, 0, x_target.shape[0])
        assert_axis_dimension(target, 1, x_target.shape[1])

        target = self._decoder(target)
        return self._family(target)
