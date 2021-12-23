import abc

import chex
import jax.numpy as np


# pylint: disable=too-few-public-methods
class Attention(abc.ABC):
    """
    Abstract attention base class
    """

    @abc.abstractmethod
    def __call__(self, key: np.ndarray, value: np.ndarray, query: np.ndarray):
        pass

    @staticmethod
    def _check_dimensions(
        key: np.ndarray, value: np.ndarray, query: np.ndarray
    ):
        chex.assert_rank([key, value, query], 3)
        chex.assert_axis_dimension(key, 0, value.shape[0])
        chex.assert_axis_dimension(query, 0, value.shape[0])
        chex.assert_axis_dimension(key, 1, value.shape[1])
        chex.assert_axis_dimension(key, 2, query.shape[2])

    @staticmethod
    def _check_return_dimension(
        rep: np.ndarray, value: np.ndarray, query: np.ndarray
    ):
        chex.assert_axis_dimension(rep, 0, value.shape[0])
        chex.assert_axis_dimension(rep, 1, query.shape[1])
        chex.assert_axis_dimension(rep, 2, value.shape[2])
