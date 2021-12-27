from typing import Optional

import haiku as hk
import jax.numpy as np

from ramsey._src.neural_process.attention.attention import Attention


class MultiHeadAttention(Attention):
    """
    Multi-head attention

    As described in [1]

    References
    ----------
    .. [1] Vaswani, Ashish, et al. "Attention is all you need."
       Advances in Neural Information Processing Systems. 2017.
    """

    def __init__(
        self, num_heads, head_size, embedding: Optional[hk.Module] = None
    ):
        """
        Instantiates a multi-head attender

        Parameters
        ----------
        num_heads: int
            number of heads
        head_size: int
            size of the heads for keys, values and queries
        embedding: hk.Module
            neural network module to embed keys and queries before attention
        """

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
