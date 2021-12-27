from typing import Tuple

import haiku as hk
import jax.numpy as np
from chex import assert_axis_dimension

from ramsey._src.neural_process.attention.attention import Attention
from ramsey.family import Family, Gaussian

__all__ = ["ANP"]


from ramsey._src.neural_process.neural_process import NP


# pylint: disable=too-many-instance-attributes,duplicate-code
class ANP(NP):
    """
    An attentive neural process

    Implements the core structure of an attentive neural process [1], i.e.,
    a deterministic encoder, a latent encoder, and a decoder with a
    cross-attention module.

    References
    ----------
    .. [1] Kim, Hyunjik, et al. "Attentive Neural Processes."
       International Conference on Learning Representations. 2019.
    """

    def __init__(
        self,
        decoder: hk.Module,
        latent_encoder: Tuple[hk.Module, hk.Module],
        deterministic_encoder: Tuple[hk.Module, Attention],
        family: Family = Gaussian(),
    ):
        """
        Instantiates an attentive neural process

        Parameters
        ----------
        decoder: hk.Module
            the decoder can be any network, but is typically an MLP. Note
            that the _last_ layer of the decoder needs to
            have twice the number of nodes as the data you try to model
        latent_encoder: Tuple[hk.Module, hk.Module]
            a tuple of two `hk.Module`s. The latent encoder can be any network,
            but is typically an MLP. The first element of the tuple is a neural
            network used before the aggregation step, while the second element
            of the tuple encodes is a neural network used to
            compute mean(s) and standard deviation(s) of the latent Gaussian.
        deterministic_encoder: Tuple[hk.Module, Attention]
            a tuple of a `hk.Module` and an Attention object. The deterministic
            encoder can be any network, but is typically an MLP
        family: Family
            distributional family of the response variable
        """

        super().__init__(
            decoder, latent_encoder, deterministic_encoder[0], family
        )
        self._deterministic_cross_attention = deterministic_encoder[1]

    @staticmethod
    def _concat_and_tile(z_deterministic, z_latent, num_observations):
        if z_latent.shape[1] == 1:
            z_latent = np.tile(z_latent, [1, num_observations, 1])
        representation = np.concatenate([z_deterministic, z_latent], axis=-1)
        assert_axis_dimension(representation, 1, num_observations)
        return representation

    def _encode_deterministic(
        self,
        x_context: np.ndarray,
        y_context: np.ndarray,
        x_target: np.ndarray,
    ):
        xy_context = np.concatenate([x_context, y_context], axis=-1)
        z_deterministic = self._deterministic_encoder(xy_context)
        z_deterministic = self._deterministic_cross_attention(
            x_context, z_deterministic, x_target
        )
        return z_deterministic
