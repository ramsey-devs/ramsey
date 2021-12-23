import haiku as hk
import jax.numpy as np
from jax import nn

from pax._src.neural_process.attention.attention import Attention


class DotProductAttention(Attention, hk.Module):
    """
    Dot-product attention
    """

    def __init__(self):
        super().__init__()

    def __call__(self, key: np.ndarray, value: np.ndarray, query: np.ndarray):
        self._check_dimensions(key, value, query)
        _, _, d_k = query.shape
        scale = np.sqrt(d_k)
        weights = np.einsum("bik,bjk->bij", query, key) / scale
        weights = nn.softmax(weights)
        rep = np.einsum("bik,bkj->bij", weights, value)
        self._check_return_dimension(rep, value, query)
        return rep
