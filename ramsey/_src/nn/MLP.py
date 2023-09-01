from typing import Callable, Iterable

import jax
from flax import linen as nn
from flax.linen import initializers
from flax.linen.linear import default_kernel_init
from jax import numpy as jnp


class MLP(nn.Module):
    output_sizes: Iterable[int]
    dropout: float = None
    kernel_init: Callable = default_kernel_init
    bias_init: Callable = initializers.zeros_init()
    use_bias: bool = True
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu
    activate_final: bool = False

    def setup(self):
        output_sizes = tuple(self.output_sizes)
        layers = []
        for index, output_size in enumerate(output_sizes):
            layers.append(
                nn.Dense(
                    features=output_size,
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                    use_bias=self.use_bias,
                    name="linear_%d" % index,
                )
            )
        self.layers = tuple(layers)
        if self.dropout is not None:
            self.dropout_layer = nn.Dropout(self.dropout)

    def __call__(self, inputs: jnp.ndarray, is_training=False):
        num_layers = len(self.layers)
        out = inputs
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < (num_layers - 1) or self.activate_final:
                if self.dropout is not None:
                    out = self.dropout_layer(out, deterministic=not is_training)
                out = self.activation(out)
        return out
