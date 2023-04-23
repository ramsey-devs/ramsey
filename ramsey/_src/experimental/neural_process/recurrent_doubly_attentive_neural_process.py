from typing import Tuple

import haiku as hk
import jax.numpy as np
from chex import assert_axis_dimension, assert_rank

from ramsey import DANP
from ramsey.attention import Attention
from ramsey.family import Family, Gaussian

__all__ = ["RDANP"]


# pylint: disable=too-many-instance-attributes,duplicate-code
class RDANP(DANP):
    """
    A recurrent doubly attentive neural process

    Implements the core structure of a recurrent attentive neural process
    cross-attention module.
    """

    def __init__(
        self,
        decoder: hk.DeepRNN,
        latent_encoder: Tuple[hk.Module, Attention, hk.Module],
        deterministic_encoder: Tuple[hk.Module, Attention, Attention],
        family: Family = Gaussian(),
    ):
        """
        Instantiates a recurrent attentive neural process

        Parameters
        ----------
        decoder: hk.DeepRNN
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

        super().__init__(decoder, latent_encoder, deterministic_encoder, family)

    def _decode(
        self,
        representation: np.ndarray,
        x_target: np.ndarray,
        y: np.ndarray,  # pylint: disable=invalid-name
    ):
        target = np.concatenate([representation, x_target], axis=-1)
        assert_rank(target, 3)
        assert_axis_dimension(target, 0, x_target.shape[0])
        assert_axis_dimension(target, 1, x_target.shape[1])

        n_batch, _, _ = target.shape
        target, _ = hk.dynamic_unroll(
            self._decoder,
            target,
            self._decoder.initial_state(n_batch),
            time_major=False,
        )
        family = self._family(target)
        self._check_posterior_predictive_axis(family, x_target, y)
        return family
