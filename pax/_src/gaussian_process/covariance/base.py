from typing import Callable

import jax.numpy as np
from jax import vmap


def squared_distance(
    x: np.ndarray,  # pylint: disable=invalid-name
    y: np.ndarray,  # pylint: disable=invalid-name
):
    return np.sum((x - y) ** 2)


def covariance(
    kernel: Callable,
    params: dict,
    x: np.ndarray,  # pylint: disable=invalid-name
    y: np.ndarray,  # pylint: disable=invalid-name
) -> np.ndarray:
    return vmap(lambda x1: vmap(lambda y1: kernel(x1, y1, **params))(x))(y)
