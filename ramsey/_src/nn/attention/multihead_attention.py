import functools
from collections.abc import Callable

from flax import linen as nn
from flax.linen import dot_product_attention, initializers
from flax.linen.linear import (
    DenseGeneral,
    DotGeneralT,
    PrecisionLike,
    default_kernel_init,
)
from flax.linen.module import merge_param
from jax import Array, lax
from jax import numpy as jnp

from ramsey._src.nn.attention.attention import Attention


# ruff: noqa: PLR0913
class MultiHeadAttention(Attention):
    """Multi-head attention.

    As described in [1].

    Attributes
    ----------
    num_heads: int
       number of heads
    head_size: int
       size of the heads for keys, values and queries
    embedding: flax.linen.Module
       neural network module to embed keys and queries before attention

    References
    ----------
    .. [1] Vaswani, Ashish, et al. "Attention is all you need."
       Advances in Neural Information Processing Systems. 2017.
    """

    num_heads: int
    head_size: int
    embedding: nn.Module | None

    def setup(self) -> None:
        """Construct the networks."""
        self._attention = _MultiHeadAttention(
            num_heads=self.num_heads,
            qkv_features=self.head_size * self.num_heads,
            out_features=self.head_size,
        )

    @nn.compact
    def __call__(self, key: Array, value: Array, query: Array) -> Array:
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
        rep = self._attention(query, key, value)
        self._check_return_dimension(rep, value, query)
        return rep


# ruff: noqa: E501
class _MultiHeadAttention(nn.Module):
    num_heads: int
    dtype = None
    param_dtype = jnp.float32
    qkv_features: int | None = None
    out_features: int | None = None
    broadcast_dropout: bool = True
    dropout_rate: float = 0.0
    deterministic: bool | None = None
    precision: PrecisionLike = None
    kernel_init: Callable = default_kernel_init
    bias_init: Callable = initializers.zeros_init()
    use_bias: bool = True
    attention_fn: Callable[..., Array] = dot_product_attention
    decode: bool = False
    qkv_dot_general: DotGeneralT = lax.dot_general
    out_dot_general: DotGeneralT = lax.dot_general

    @nn.compact
    def __call__(
        self,
        query: Array,
        key: Array,
        value: Array,
        mask: Array | None = None,
        deterministic: bool | None = None,
    ) -> Array:
        features = self.out_features or query.shape[-1]
        qkv_features = self.qkv_features or query.shape[-1]
        assert (
            qkv_features % self.num_heads == 0
        ), "Memory dimension must be divisible by number of heads."
        head_dim = qkv_features // self.num_heads

        dense = functools.partial(
            DenseGeneral,
            axis=-1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            features=(self.num_heads, head_dim),
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            precision=self.precision,
            dot_general=self.qkv_dot_general,
        )

        query, key, value = (
            dense(name="query")(query),
            dense(name="key")(key),
            dense(name="value")(value),
        )

        dropout_rng = None
        if self.dropout_rate > 0.0:
            m_deterministic = merge_param(
                "deterministic", self.deterministic, deterministic
            )
            if not m_deterministic:
                dropout_rng = self.make_rng("dropout")
        else:
            m_deterministic = True

        # apply attention
        x = self.attention_fn(
            query,
            key,
            value,
            mask=mask,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout_rate,
            broadcast_dropout=self.broadcast_dropout,
            deterministic=m_deterministic,
            dtype=self.dtype,
            precision=self.precision,
        )  # pytype: disable=wrong-keyword-args
        # back to the original inputs dimensions
        out = DenseGeneral(
            features=features,
            axis=(-2, -1),
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            dot_general=self.out_dot_general,
            name="out",  # type: ignore[call-arg]
        )(x)
        return out
