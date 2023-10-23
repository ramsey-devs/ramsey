from typing import Optional, Tuple

from chex import assert_axis_dimension, assert_rank
from flax import linen as nn
from jax import Array
from jax import numpy as jnp

from ramsey import ANP
from ramsey.family import Family, Gaussian
from ramsey.nn import Attention

__all__ = ["RANP"]


# pylint: disable=too-many-instance-attributes,duplicate-code
class RANP(ANP):
    """A recurrent attentive neural process.

    Implements the core structure of a recurrent attentive neural process
    cross-attention module.

    Attributes
    ----------
    decoder: nn.Sequential
        the decoder can be any network, but is typically an MLP. Note
        that the _last_ layer of the decoder needs to
        have twice the number of nodes as the data you try to model
    latent_encoder: Optional[Tuple[nn.Module, nn.Module]]
        a tuple of two `nn.Module`s. The latent encoder can be any network,
        but is typically an MLP. The first element of the tuple is a neural
        network used before the aggregation step, while the second element
        of the tuple encodes is a neural network used to
        compute mean(s) and standard deviation(s) of the latent Gaussian.
    deterministic_encoder: Optional[Tuple[nn.Module, Attention]]
        a tuple of a `nn.Module` and an Attention object. The deterministic
        encoder can be any network, but is typically an MLP
    family: Family
        distributional family of the response variable
    """

    decoder: nn.Module
    latent_encoder: Optional[Tuple[nn.Module, nn.Module]] = None
    deterministic_encoder: Optional[Tuple[nn.Module, Attention]] = None
    family: Family = Gaussian()

    def setup(self):
        """Construct all networks."""
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
        if self.deterministic_encoder is not None:
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
