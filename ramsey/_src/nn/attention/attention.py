from typing import Optional

import chex
from flax import linen as nn
from jax import Array


# pylint: disable=too-few-public-methods
class Attention(nn.Module):
    """Abstract attention base class.

    Can be used for designing attention modules.

    Attributes
    ----------
    embedding: Optional[nn.Module]
        an optional embedding network that embeds keys and queries
    """

    embedding: Optional[nn.Module]

    @nn.compact
    def __call__(self, key: Array, value: Array, query: Array):
        """Pay attention to some key-value pairs."""
        self._check_dimensions(key, value, query)
        if self.embedding is not None:
            key, query = self.embedding(key), self.embedding(query)
        return key, value, query

    @staticmethod
    def _check_dimensions(key: Array, value: Array, query: Array):
        chex.assert_rank([key, value, query], 3)
        chex.assert_axis_dimension(key, 0, value.shape[0])
        chex.assert_axis_dimension(key, 1, value.shape[1])
        chex.assert_axis_dimension(key, 2, query.shape[2])
        chex.assert_axis_dimension(query, 0, value.shape[0])

    @staticmethod
    def _check_return_dimension(rep: Array, value: Array, query: Array):
        chex.assert_axis_dimension(rep, 0, value.shape[0])
        chex.assert_axis_dimension(rep, 1, query.shape[1])
