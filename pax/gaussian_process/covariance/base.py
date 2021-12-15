from typing import Callable
from jax import vmap
import jax.numpy as np


def squared_distance(x: np.DeviceArray, y: np.DeviceArray):
    return np.sum((x - y) ** 2)


def covariance(
    kernel: Callable,
    params: dict,
    x: np.DeviceArray,
    y: np.DeviceArray = None
) -> np.DeviceArray:
    return vmap(lambda x1: vmap(lambda y1: kernel(x1, y1, **params))(x))(y)
