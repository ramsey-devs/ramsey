import functools
from typing import (Any, Callable, Optional, Tuple)

from flax import linen as nn
from flax.linen import initializers, dot_product_attention
from flax.linen.linear import DenseGeneral
from flax.linen.linear import DotGeneralT
from flax.linen.linear import PrecisionLike
from flax.linen.linear import default_kernel_init
from flax.linen.module import merge_param
from jax import lax

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any
from jax import numpy as jnp, Array

from ramsey._src.attention.attention import Attention


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
            self, num_heads, head_size, embedding: Optional[nn.Module] = None
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
        self._attention = _MultiHeadAttention(
            num_heads=num_heads,
            qkv_features=head_size * num_heads,
            out_features=head_size,
        )

    def __call__(self, key: Array, value: Array, query: Array):
        key, value, query = super().__call__(key, value, query)
        rep = self._attention(query, key, value)
        self._check_return_dimension(rep, value, query)
        return rep


class _MultiHeadAttention(nn.Module):
    num_heads: int
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    qkv_features: Optional[int] = None
    out_features: Optional[int] = None
    broadcast_dropout: bool = True
    dropout_rate: float = 0.
    deterministic: Optional[bool] = None
    precision: PrecisionLike = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[
        [PRNGKey, Shape, Dtype], Array] = initializers.zeros_init()
    use_bias: bool = True
    attention_fn: Callable[..., Array] = dot_product_attention
    decode: bool = False
    qkv_dot_general: DotGeneralT = lax.dot_general
    out_dot_general: DotGeneralT = lax.dot_general

    @nn.compact
    def __call__(self,
                 query: Array,
                 key: Array,
                 value: Array,
                 mask: Optional[Array] = None,
                 deterministic: Optional[bool] = None):
        features = self.out_features or query.shape[-1]
        qkv_features = self.qkv_features or query.shape[-1]
        assert qkv_features % self.num_heads == 0, (
            'Memory dimension must be divisible by number of heads.')
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

        query, key, value = (dense(name='query')(query),
                             dense(name='key')(key),
                             dense(name='value')(value))

        dropout_rng = None
        if self.dropout_rate > 0.:
            m_deterministic = merge_param('deterministic', self.deterministic,
                                          deterministic)
            if not m_deterministic:
                dropout_rng = self.make_rng('dropout')
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
            precision=self.precision)  # pytype: disable=wrong-keyword-args
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
            name='out',  # type: ignore[call-arg]
        )(x)
        return out
