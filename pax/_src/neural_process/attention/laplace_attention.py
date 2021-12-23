import jax.numpy as np
from jax import nn

from pax._src.neural_process.attention.attention import Attention


# pylint: disable=too-few-public-methods
class LaplaceAttention(Attention):
    """
    Laplace kernel attention
    """

    def __init__(self, scale, normalize):
        self._scale = scale
        self._normalize = normalize
        if normalize:
            self._weight_fn = nn.softmax
        else:
            self._weight_fn = lambda x: 1 + nn.tanh(x)

    def __call__(self, key: np.ndarray, value: np.ndarray, query: np.ndarray):
        self._check_dimensions(key, value, query)
        key = np.expand_dims(key, axis=1)
        query = np.expand_dims(query, axis=2)
        weights = -np.abs((key - query) / self._scale)
        weights = np.sum(weights, axis=-1)
        weights = self._weight_fn(weights)
        rep = np.einsum("bik,bkj->bij", weights, value)
        self._check_return_dimension(rep, value, query)
        return rep
