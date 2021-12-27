from typing import Tuple

import haiku as hk
import jax.numpy as np

from ramsey.family import Family, Gaussian

__all__ = ["DANP"]

from ramsey._src.neural_process.attention.attention import Attention
from ramsey._src.neural_process.attentive_neural_process import ANP


# pylint: disable=too-many-instance-attributes
class DANP(ANP):
    """
    A doubly-attentive neural process

    Implements the core structure of a 'doubly-attentive' neural process [1],
    i.e., a deterministic encoder, a latent encoder with self-attention module,
    and a decoder with both self- and cross-attention modules.

    References
    ----------
    .. [1] Kim, Hyunjik, et al. "Attentive Neural Processes."
       International Conference on Learning Representations. 2019.
    """

    def __init__(
        self,
        decoder: hk.Module,
        latent_encoder: Tuple[hk.Module, Attention, hk.Module],
        deterministic_encoder: Tuple[hk.Module, Attention, Attention],
        family: Family = Gaussian(),
    ):
        """
        Instantiates a doubly-attentive neural process

        Parameters
        ----------
        decoder: hk.Module
            the decoder can be any network, but is typically an MLP. Note
            that the _last_ layer of the decoder needs to
            have twice the number of nodes as the data you try to model
        latent_encoder: Tuple[hk.Module, Attention, hk.Module]
            a tuple of two `hk.Module`s and an attention object. The first and
            last elements are the usual modules required for a neural process,
            the attention object computes self-attention before the aggregation
        deterministic_encoder: Tuple[hk.Module, Attention, Attention]
            ea tuple of a `hk.Module` and an Attention object. The first
            `attention` object is used for self-attention, the second one
            is used for cross-attention
        family: Family
            distributional family of the response variable
        """

        super().__init__(
            decoder,
            (latent_encoder[0], latent_encoder[2]),
            (deterministic_encoder[0], deterministic_encoder[2]),
            family,
        )
        self._latent_self_attention = latent_encoder[1]
        self._deterministic_self_attention = deterministic_encoder[1]

    def _encode_latent(self, x_context: np.ndarray, y_context: np.ndarray):
        xy_context = np.concatenate([x_context, y_context], axis=-1)
        z_latent = self._latent_encoder(xy_context)
        z_latent = self._latent_self_attention(z_latent, z_latent, z_latent)
        return self._encode_latent_gaussian(z_latent)

    def _encode_deterministic(
        self,
        x_context: np.ndarray,
        y_context: np.ndarray,
        x_target: np.ndarray,
    ):
        xy_context = np.concatenate([x_context, y_context], axis=-1)
        z_deterministic = self._deterministic_encoder(xy_context)
        z_deterministic = self._deterministic_self_attention(
            z_deterministic, z_deterministic, z_deterministic
        )
        z_deterministic = self._deterministic_cross_attention(
            x_context, z_deterministic, x_target
        )
        return z_deterministic
