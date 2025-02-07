import dataclasses
from collections.abc import Callable

import flax
import jax
from flax import nnx
from flax.nnx import rnglib

from ramsey._src.nn.attention.attention import Attention


@dataclasses.dataclass
class MultiHeadAttention(Attention):
  """Multi-head attention.

  As described in [1].

  Args:
      in_features: int
      num_heads: number of heads
      embedding: neural network module to embed keys and queries before
          attention
      rngs: a rnglib.Rngs object for random seeds
  """

  def __init__(
    self,
    in_features: int,
    num_heads: int,
    embedding: flax.nnx.Module | Callable = lambda x: x,
    *,
    rngs: rnglib.Rngs | None = None,
  ):
    """Construct the networks."""
    super().__init__(embedding)
    self._attention = nnx.MultiHeadAttention(
      in_features=in_features,
      num_heads=num_heads,
      decode=False,
      rngs=rngs,
    )

  def __call__(
    self,
    key: jax.Array,
    value: jax.Array,
    query: jax.Array,
    *,
    rngs: rnglib.Rngs | None = None,
  ) -> jax.Array:
    """Apply attention to the query.

    Args:
      key: the key :)
      value: the value :)
      query: the query :)
      rngs: a nnx random key

    Returns:
      returns attended query
    """
    key, value, query = super().__call__(key, value, query)
    rep = self._attention(query, key, value, rngs=rngs)
    self._check_return_dimension(rep, value, query)
    return rep
