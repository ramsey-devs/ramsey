from jax import numpy as jnp


def squared_distance(
    x: jnp.ndarray,  # pylint: disable=invalid-name
    y: jnp.ndarray,  # pylint: disable=invalid-name
):
    return jnp.sum((x - y) ** 2)
