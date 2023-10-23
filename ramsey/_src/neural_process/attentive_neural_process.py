from typing import Optional

from chex import assert_axis_dimension
from flax import linen as nn
from jax import numpy as jnp

from ramsey._src.family import Family, Gaussian
from ramsey._src.neural_process.neural_process import NP

__all__ = ["ANP"]


# pylint: disable=too-many-instance-attributes,duplicate-code
# pylint: disable=unpacking-non-sequence,
class ANP(NP):
    """An attentive neural process.

    Implements the core structure of an attentive neural process [1], i.e.,
    a deterministic encoder, a latent encoder, and a decoder with a
    cross-attention module.

    Attributes
    ----------
    decoder: flax.linen.Module
        the decoder can be any network, but is typically an MLP. Note
        that the _last_ layer of the decoder needs to
        have twice the number of nodes as the data you try to model
    latent_encoders: Optional[Tuple[flax.linen.Module, flax.linen.Module]]
        a tuple of two `flax.linen.Module`s. The latent encoder can be any
        network, but is typically an MLP. The first element of the tuple is a
        neural network used before the aggregation step, while the second
        element of the tuple encodes is a neural network used to
        compute mean(s) and standard deviation(s) of the latent Gaussian.
    deterministic_encoder: Optional[Tuple[flax.linen.Module, Attention]]
        a tuple of a `flax.linen.Module` and an Attention object.
        The deterministic encoder can be any network, but is typically an MLP
    family: Family
        distributional family of the response variable

    References
    ----------
    .. [1] Kim, Hyunjik, et al. "Attentive Neural Processes."
        International Conference on Learning Representations. 2019.
    """

    decoder: nn.Module
    latent_encoder: Optional[nn.Module] = None
    deterministic_encoder: Optional[nn.Module] = None
    family: Family = Gaussian()

    def setup(self):
        """Construct the neural process parameters."""
        if self.latent_encoder is None and self.deterministic_encoder is None:
            raise ValueError(
                "either latent or deterministic encoder needs to be set"
            )
        self._decoder = self.decoder
        if self.latent_encoder is not None:
            (self._latent_encoder, self._latent_variable_encoder) = (
                self.latent_encoder[0],
                self.latent_encoder[1],
            )
        self._deterministic_encoder = self.deterministic_encoder[0]
        self._deterministic_cross_attention = self.deterministic_encoder[1]
        self._family = self.family

    @staticmethod
    def _concat_and_tile(z_deterministic, z_latent, num_observations):
        if z_latent is not None:
            if z_latent.shape[1] == 1:
                z_latent = jnp.tile(z_latent, [1, num_observations, 1])
            representation = jnp.concatenate(
                [z_deterministic, z_latent], axis=-1
            )
        else:
            representation = z_deterministic
        assert_axis_dimension(representation, 1, num_observations)
        return representation

    def _encode_deterministic(self, x_context, y_context, x_target):
        xy_context = jnp.concatenate([x_context, y_context], axis=-1)
        z_deterministic = self._deterministic_encoder(xy_context)
        z_deterministic = self._deterministic_cross_attention(
            x_context, z_deterministic, x_target
        )
        return z_deterministic
