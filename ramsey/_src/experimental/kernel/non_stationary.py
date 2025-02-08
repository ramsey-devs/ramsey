import jax
from flax import nnx
from flax.nnx import rnglib
from flax.typing import Dtype
from jax import numpy as jnp

from ramsey._src.experimental.kernel.base import Kernel


class Linear(Kernel, nnx.Module):
  """Linear covariance function.

  Args:
    active_dims: the indexes of the dimensions the kernel acts upon
    sigma_b_init: an initializer object from Flax or None
    sigma_v_init: an initializer object from Flax or None
    offset_init: an initializer object from Flax or None
    rngs: a random seed generator
  """

  def __init__(
    self,
    active_dims: list | None = None,
    *,
    sigma_b_init: nnx.initializers.Initializer = nnx.initializers.uniform(),
    sigma_v_init: nnx.initializers.Initializer = nnx.initializers.uniform(),
    offset_init: nnx.initializers.Initializer = nnx.initializers.zeros_init(),
    param_dtype: Dtype = jnp.float32,
    rngs: rnglib.Rngs,
  ):
    self._active_dims = (
      active_dims if isinstance(active_dims, list) else slice(active_dims)
    )

    self.log_sigma_b = nnx.Param(sigma_b_init(rngs.params(), (), param_dtype))
    self.log_sigma_v = nnx.Param(sigma_v_init(rngs.params(), (), param_dtype))
    self.offset = nnx.Param(offset_init(rngs.params(), (), param_dtype))

  def __call__(self, x1: jax.Array, x2: jax.Array = None):
    """Call the covariance function."""
    if x2 is None:
      x2 = x1
    cov = linear(
      x1[..., self._active_dims],
      x2[..., self._active_dims],
      jnp.exp(self.log_sigma_b.value),
      jnp.exp(self.log_sigma_v.value),
      self.offset.value,
    )
    return cov


def linear(
  x1: jax.Array, x2: jax.Array, sigma_b: float, sigma_v: float, offset: float
):
  r"""Linear convariance function.

  Args:
    x1: :math:`n x p`-dimensional set of data points
    x2: :math:`m x p`-dimensional set of data points
    sigma_b: the standard deviation of the kernel function
    sigma_v: the standard deviation of the kernel function
    offset: float

  Returns:
      returns a :math:`n x m`-dimensional Gram matrix
  """

  def _linear(x1, x2, sigma_b, sigma_v, offset):
    x_e = x1 - offset
    y_e = x2 - offset
    x_e = jnp.expand_dims(x_e, 1)
    y_e = jnp.expand_dims(y_e, 0)
    d = jnp.sum(x_e * y_e, axis=2)
    K = sigma_v**2 * d + sigma_b**2
    return K

  return _linear(x1, x2, sigma_b, sigma_v, offset)
