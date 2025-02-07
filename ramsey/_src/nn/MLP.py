import dataclasses
from collections.abc import Callable, Iterable

import jax
from flax import nnx
from flax.nnx import rnglib


@dataclasses.dataclass
class MLP(nnx.Module):
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

  input_size: int
  output_sizes: Iterable[int]
  dropout: float | None = None
  kernel_init: nnx.initializers.Initializer = nnx.initializers.lecun_normal()
  bias_init: nnx.initializers.Initializer = nnx.initializers.zeros_init()
  use_bias: bool = True
  activation: Callable = jax.nn.relu
  activate_final: bool = False
  rngs: rnglib.Rngs | None = None

  def __post_init__(self):
    """Construct all networks."""
    output_sizes = (self.input_size,) + tuple(self.output_sizes)
    layers = []
    for index, (din, dout) in enumerate(
      zip(output_sizes[:-1], output_sizes[1:])
    ):
      layers.append(
        nnx.Linear(
          in_features=din,
          out_features=dout,
          kernel_init=self.kernel_init,
          bias_init=self.bias_init,
          use_bias=self.use_bias,
          rngs=self.rngs,
        )
      )
    self.layers = tuple(layers)
    if self.dropout is not None:
      self.dropout_layer = nnx.Dropout(self.dropout)

  def __call__(
    self,
    inputs: jax.Array,
    is_training: bool = False,
    *,
    rngs: rnglib.Rngs | None = None,
  ) -> jax.Array:
    """Transform the inputs through the MLP.

    Args:
      inputs: input data of dimension
        (*batch_dims, spatial_dims..., feature_dims)
      is_training: if true, uses training mode (i.e., dropout)
      rngs: a random seed generator

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
