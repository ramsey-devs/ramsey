from collections.abc import Callable

import flax
import jax
from flax import linen as nn

from ramsey._src.nn.attention.attention import Attention


# ruff: noqa: PLR091
class MultiHeadAttention(Attention):
  """Multi-head attention.

  Args:
    num_heads: number of heads
    embedding: neural network module to embed keys and queries before
      attention
  """

  num_heads: int
  embedding: flax.linen.Module | Callable

  def setup(self):
    """Construct the networks."""
    self._attention = nn.MultiHeadAttention(num_heads=self.num_heads)

  def __call__(
    self,
    key: jax.Array,
    value: jax.Array,
    query: jax.Array,
  ) -> jax.Array:
    """Apply attention to the query.

    Args:
      key: the key :)
      value: the value :)
      query: the query :)

    Returns:
      returns attended query
    """
    key, value, query = super().__call__(key, value, query)
    rep = self._attention(query, key, value)
    self._check_return_dimension(rep, value, query)
    return rep
