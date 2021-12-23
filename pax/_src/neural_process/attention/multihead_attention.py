import haiku as hk
import jax.numpy as np

from pax._src.neural_process.attention.attention import Attention


class MultiHeadAttention(Attention, hk.Module):
    """
    Multi-head attention
    """

    def __init__(self, num_heads, key_size, value_size, w_init_scale):
        super().__init__()
        self._attention = hk.MultiHeadAttention(
            num_heads,
            key_size=key_size,
            value_size=value_size,
            w_init_scale=w_init_scale,
        )

    def __call__(self, key: np.ndarray, value: np.ndarray, query: np.ndarray):
        self._check_dimensions(key, value, query)
        rep = self._attention(query, key, value)
        self._check_return_dimension(rep, value, query)
        return rep
