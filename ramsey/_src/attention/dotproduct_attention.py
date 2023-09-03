import jax
from jax import Array
from jax import numpy as jnp

from ramsey._src.attention.attention import Attention


class DotProductAttention(Attention):
    """
    Dot-product attention
    """

    def __call__(self, key: Array, value: Array, query: Array):
        key, value, query = super().__call__(key, value, query)
        _, _, d_k = query.shape
        scale = jnp.sqrt(d_k)
        weights = jnp.einsum("bik,bjk->bij", query, key) / scale
        weights = jax.nn.softmax(weights)
        rep = jnp.einsum("bik,bkj->bij", weights, value)
        self._check_return_dimension(rep, value, query)
        return rep
