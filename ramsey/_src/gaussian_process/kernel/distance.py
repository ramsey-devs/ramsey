

from jax import numpy as jnp, Array


def squared_distance(
    x: Array,  # pylint: disable=invalid-name
    y: Array,  # pylint: disable=invalid-name
):
    return jnp.sum((x - y) ** 2)
