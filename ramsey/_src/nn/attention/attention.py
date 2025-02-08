import chex
import jax
from flax import nnx


class Attention(nnx.Module):
  """Abstract attention base class.

  Can be used for designing attention modules.

  Args:
      embedding: an optional embedding network that embeds keys and queries
  """

  def __init__(self, embedding: nnx.Module):
    self._embedding = embedding

  def __call__(
    self, key: jax.Array, value: jax.Array, query: jax.Array
  ) -> jax.Array:
    """Pay attention to some key-value pairs."""
    self._check_dimensions(key, value, query)
    key, query = self._embedding(key), self._embedding(query)
    return key, value, query

  @staticmethod
  def _check_dimensions(key: jax.Array, value: jax.Array, query: jax.Array):
    chex.assert_rank([key, value, query], 3)
    chex.assert_axis_dimension(key, 0, value.shape[0])
    chex.assert_axis_dimension(key, 1, value.shape[1])
    chex.assert_axis_dimension(key, 2, query.shape[2])
    chex.assert_axis_dimension(query, 0, value.shape[0])

  @staticmethod
  def _check_return_dimension(
    rep: jax.Array, value: jax.Array, query: jax.Array
  ):
    chex.assert_axis_dimension(rep, 0, value.shape[0])
    chex.assert_axis_dimension(rep, 1, query.shape[1])
