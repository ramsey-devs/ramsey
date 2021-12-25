from typing import Tuple

import haiku as hk

from pax._src.family import Family, Gaussian

__all__ = ["DANP"]

from pax._src.neural_process.attention.attention import Attention
from pax._src.neural_process.attentive_neural_process import ANP


# pylint: disable=too-many-instance-attributes
class DANP(ANP):
    """
    A doubly-attentive neural process

    Implements the core structure of a neural process, i.e., two encoders
    and a decoder, as a haiku module. Needs to be called directly within
    `hk.transform` with the respective arguments.
    """

    def __init__(
        self,
        decoder: hk.Module,
        latent_encoder: Tuple[hk.Module, Attention, hk.Module],
        deterministic_encoder: Tuple[hk.Module, Attention, Attention],
        family: Family = Gaussian(),
    ):
        """
        Instantiates a doubly-attentivee neural process

        Parameters
        ----------
        decoder: hk.Module
            either a function that wraps an `hk.Module` and calls it or a
            `hk.Module`. The decoder can be any network, but is
            typically an MLP. Note that the _last_ layer of the decoder needs to
            have twice the number of nodes as the data you try to model!
            That means if your response is univariate
        latent_encoder:  Tuple[hk.Module, hk.Module]
            a tuple of either functions that wrap `hk.Module`s and calls them or
            two `hk.Module`s. The latent encoder can be any network, but is
            typically an MLP. The first element of the tuple is a neural network
            used before the aggregation step, while the second element of
            the tuple encodes is a neural network used to
            compute mean(s) and standard deviation(s) of the latent Gaussian.
        deterministic_encoder: Union[hk.Module, None]
            either a function that wraps an `hk.Module` and calls it or a
            `hk.Module`. The deterministic encoder can be any network, but is
            typically an MLP
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
