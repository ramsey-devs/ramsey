from typing import Any

from flax import linen as nn
from jax import Array
from jax import numpy as jnp

from ramsey._src.family import Family, Gaussian
from ramsey._src.neural_process.attentive_neural_process import ANP

__all__ = ["DANP"]


# pylint: disable=too-many-instance-attributes
class DANP(ANP):
    """
    A doubly-attentive neural process.

    Implements the core structure of a 'doubly-attentive' neural process [1],
    i.e., a deterministic encoder, a latent encoder with self-attention module,
    and a decoder with both self- and cross-attention modules.

    Attributes
    ----------
    decoder: flax.linen.Module
            the decoder can be any network, but is typically an MLP. Note
            that the _last_ layer of the decoder needs to
            have twice the number of nodes as the data you try to model
    latent_encoder: Tuple[flax.linen.Module, Attention, flax.linen.Module]
        a tuple of two `flax.linen.Module`s and an attention object. The first
        and
        last elements are the usual modules required for a neural process,
        the attention object computes self-attention before the aggregation
    deterministic_encoder: Tuple[flax.linen.Module, Attention, Attention]
        ea tuple of a `flax.linen.Module` and an Attention object. The first
        `attention` object is used for self-attention, the second one
        is used for cross-attention
    family: Family
        distributional family of the response variable

    References
    ----------
    .. [1] Kim, Hyunjik, et al. "Attentive Neural Processes."
       International Conference on Learning Representations. 2019.
    """

    decoder: nn.Module
    latent_encoder: Any
    deterministic_encoder: Any
    family: Family = Gaussian()

    def setup(self):
        """Construct all networks."""
        if self.latent_encoder is None and self.deterministic_encoder is None:
            raise ValueError(
                "either latent or deterministic encoder needs to be set"
            )
        if self.latent_encoder is not None:
            (
                self._latent_encoder,
                self._latent_self_attention,
                self._latent_variable_encoder,
            ) = self.latent_encoder
        if self.deterministic_encoder is not None:
            (
                self._deterministic_encoder,
                self._deterministic_self_attention,
                self._deterministic_cross_attention,
            ) = self.deterministic_encoder
        self._decoder = self.decoder
        self._family = self.family

    def _encode_latent(self, x_context: Array, y_context: Array):
        xy_context = jnp.concatenate([x_context, y_context], axis=-1)
        z_latent = self._latent_encoder(xy_context)
        z_latent = self._latent_self_attention(z_latent, z_latent, z_latent)
        return self._encode_latent_gaussian(z_latent)

    def _encode_deterministic(
        self,
        x_context: Array,
        y_context: Array,
        x_target: Array,
    ):
        xy_context = jnp.concatenate([x_context, y_context], axis=-1)
        z_deterministic = self._deterministic_encoder(xy_context)
        z_deterministic = self._deterministic_self_attention(
            z_deterministic, z_deterministic, z_deterministic
        )
        z_deterministic = self._deterministic_cross_attention(
            x_context, z_deterministic, x_target
        )
        return z_deterministic
