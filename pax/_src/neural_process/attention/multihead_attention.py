from typing import Optional

import haiku as hk
import jax.numpy as np

from pax._src.neural_process.attention.attention import Attention


class MultiHeadAttention(Attention):
    """
    Multi-head attention
    """

    def __init__(
        self, num_heads, head_size, embedding: Optional[hk.Module] = None
    ):
        super().__init__(embedding)
        self._attention = hk.MultiHeadAttention(
            num_heads,
            key_size=head_size,
            value_size=head_size,
            w_init_scale=2.0,
        )

    def __call__(self, key: np.ndarray, value: np.ndarray, query: np.ndarray):
        key, value, query = super().__call__(key, value, query)
        rep = self._attention(query, key, value)
        self._check_return_dimension(rep, value, query)
        return rep
