import jax
from jax import numpy as jnp


def squared_distance(
  x: jax.Array,
  y: jax.Array,
):
  """Compute squared distance."""
  return jnp.sum((x - y) ** 2)
