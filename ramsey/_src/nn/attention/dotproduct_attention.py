import jax
from jax import Array
from jax import numpy as jnp

from ramsey._src.nn.attention.attention import Attention


class DotProductAttention(Attention):
    """Dot-product attention."""

    def __call__(self, key: Array, value: Array, query: Array):
        """Apply attention to the query.

        Arguments
        ---------
        key: jax.Array
            key
        value: jax.Array
            value
        query: jax.Array
            query

        Returns
        -------
        jax.Array
            returns attended query
        """
        key, value, query = super().__call__(key, value, query)
        _, _, d_k = query.shape
        scale = jnp.sqrt(d_k)
        weights = jnp.einsum("bik,bjk->bij", query, key) / scale
        weights = jax.nn.softmax(weights)
        rep = jnp.einsum("bik,bkj->bij", weights, value)
        self._check_return_dimension(rep, value, query)
        return rep
