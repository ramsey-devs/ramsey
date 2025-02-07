from collections.abc import Callable, Iterable

import jax
from flax import linen as nn
from flax.linen import initializers
from flax.linen.linear import default_kernel_init


class MLP(nn.Module):
  """A multi-layer perceptron.

  Args:
    output_sizes: number of hidden nodes per layer
    dropout: dropout rate to apply after each hidden layer
    kernel_init: initializer for weights of hidden layers
    bias_init: initializer for bias of hidden layers
    use_bias: boolean if hidden layers should use bias nodes
    activation: activation function to apply after each hidden layer
    activate_final: if true, activate last layer
    rngs: a random seed generator
  """

  output_sizes: Iterable[int]
  dropout: float | None = None
  kernel_init: initializers.Initializer = default_kernel_init
  bias_init: initializers.Initializer = initializers.zeros_init()
  use_bias: bool = True
  activation: Callable = jax.nn.relu
  activate_final: bool = False

  def setup(self):
    """Construct all networks."""
    output_sizes = tuple(self.output_sizes)
    layers = []
    for index, output_size in enumerate(output_sizes):
      layers.append(
        nn.Dense(
          features=output_size,
          kernel_init=self.kernel_init,
          bias_init=self.bias_init,
          use_bias=self.use_bias,
          name=f"linear_{index}",
        )
      )
    self.layers = tuple(layers)
    if self.dropout is not None:
      self.dropout_layer = nn.Dropout(self.dropout)

  def __call__(
    self,
    inputs: jax.Array,
    is_training: bool = False,
  ) -> jax.Array:
    """Transform the inputs through the MLP.

    Args:
      inputs: input data of dimension
        (*batch_dims, spatial_dims..., feature_dims)
      is_training: if true, uses training mode (i.e., dropout)

    Returns:
        returns the transformed inputs
    """
    num_layers = len(self.layers)
    out = inputs
    for i, layer in enumerate(self.layers):
      out = layer(out)
      if i < (num_layers - 1) or self.activate_final:
        if self.dropout is not None:
          out = self.dropout_layer(out, deterministic=not is_training)
        out = self.activation(out)
    return out
