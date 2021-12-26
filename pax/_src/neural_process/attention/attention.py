import abc
from typing import Optional

import chex
import haiku as hk
import jax.numpy as np


# pylint: disable=too-few-public-methods
class Attention(abc.ABC, hk.Module):
    """
    Abstract attention base class
    """

    def __init__(self, embedding: Optional[hk.Module]):
        super().__init__()
        self._embedding = embedding

    @abc.abstractmethod
    def __call__(self, key: np.ndarray, value: np.ndarray, query: np.ndarray):
        self._check_dimensions(key, value, query)
        if self._embedding is not None:
            key, query = self._embedding(key), self._embedding(query)
        return key, value, query

    @staticmethod
    def _check_dimensions(
        key: np.ndarray, value: np.ndarray, query: np.ndarray
    ):
        chex.assert_rank([key, value, query], 3)
        chex.assert_axis_dimension(key, 0, value.shape[0])
        chex.assert_axis_dimension(key, 1, value.shape[1])
        chex.assert_axis_dimension(key, 2, query.shape[2])
        chex.assert_axis_dimension(query, 0, value.shape[0])

    @staticmethod
    def _check_return_dimension(
        rep: np.ndarray, value: np.ndarray, query: np.ndarray
    ):
        chex.assert_axis_dimension(rep, 0, value.shape[0])
        chex.assert_axis_dimension(rep, 1, query.shape[1])
