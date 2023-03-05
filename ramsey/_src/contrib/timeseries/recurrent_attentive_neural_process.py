from typing import Tuple

import haiku as hk
import jax.numpy as np
from chex import assert_axis_dimension, assert_rank

from ramsey import ANP
from ramsey.attention import Attention
from ramsey.family import Family, Gaussian

__all__ = ["RANP"]


# pylint: disable=too-many-instance-attributes,duplicate-code
class RANP(ANP):
    """
    A recurrent attentive neural process

    Implements the core structure of a recurrent attentive neural process
    cross-attention module.
    """

    def __init__(
        self,
        decoder: hk.DeepRNN,
        latent_encoder: Tuple[hk.Module, hk.Module],
        deterministic_encoder: Tuple[hk.Module, Attention],
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

        _, num_observations, _ = target.shape
        target, _ = hk.dynamic_unroll(
            self._decoder, target, self._decoder.initial_state(num_observations)
        )
        return self._family(target)
