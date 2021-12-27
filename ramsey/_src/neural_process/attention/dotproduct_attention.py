import jax.numpy as np
from jax import nn

from ramsey._src.neural_process.attention.attention import Attention


class DotProductAttention(Attention):
    """
    Dot-product attention
    """

    def __call__(self, key: np.ndarray, value: np.ndarray, query: np.ndarray):
        key, value, query = super().__call__(key, value, query)
        _, _, d_k = query.shape
        scale = np.sqrt(d_k)
        weights = np.einsum("bik,bjk->bij", query, key) / scale
        weights = nn.softmax(weights)
        rep = np.einsum("bik,bkj->bij", weights, value)
        self._check_return_dimension(rep, value, query)
        return rep
