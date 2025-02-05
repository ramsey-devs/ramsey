import jax
from flax import nnx
from flax.nnx import rnglib
from flax.typing import Dtype
from jax import numpy as jnp

from ramsey._src.experimental.kernel.base import Kernel


# pylint: disable=invalid-name
class Periodic(Kernel, nnx.Module):
  """Periodic covariance function.

  Args:
    period: the period of the periodic kernel
    active_dims: either None or a list of integers.
      Specified the dimensions of the data on which the kernel operates on
    rho_init: an initializer object
    sigma_init: an initializer object from Haiku or None
    param_dtype : parameter type
    rngs: a random seed generator
  """

  def __init__(
    self,
    period,
    active_dims: list | None = None,
    *,
    rho_init: nnx.initializers.Initializer = nnx.initializers.constant(
      jnp.log(1.0)
    ),
    sigma_init: nnx.initializers.Initializer = nnx.initializers.constant(
      jnp.log(1.0)
    ),
    param_dtype: Dtype = jnp.float32,
    rngs: rnglib.Rngs,
  ):
    self.period = period
    self.log_rho = nnx.Param(rho_init(rngs.params(), (), param_dtype))
    self.log_sigma = nnx.Param(sigma_init(rngs.params(), (), param_dtype))

    self._active_dims = (
      active_dims if isinstance(active_dims, list) else slice(active_dims)
    )

  def __call__(self, x1: jax.Array, x2: jax.Array = None):
    """Call the covariance function."""
    if x2 is None:
      x2 = x1
    cov = periodic(
      x1[..., self._active_dims],
      x2[..., self._active_dims],
      self.period,
      jnp.exp(self.log_sigma),
      jnp.exp(self.log_rho),
    )
    return cov


class ExponentiatedQuadratic(Kernel, nnx.Module):
  """Exponentiated quadratic covariance function.

  Args:
    active_dims: either None or a list of integers. Specified the dimensions
      of the data on which the kernel operates on
    rho_init: Optional[Initializer]
      an initializer object from Haiku or None
    sigma_init: Optional[Initializer]
      an initializer object from Haiku or None
    param_dtype : parameter type
    rngs: a random seed generator
  """

  def __init__(
    self,
    active_dims: list | None = None,
    *,
    rho_init: nnx.initializers.Initializer = nnx.initializers.constant(
      jnp.log(1.0)
    ),
    sigma_init: nnx.initializers.Initializer = nnx.initializers.constant(
      jnp.log(1.0)
    ),
    param_dtype: Dtype = jnp.float32,
    rngs: rnglib.Rngs,
  ):
    self.log_rho = nnx.Param(rho_init(rngs.params(), (), param_dtype))
    self.log_sigma = nnx.Param(sigma_init(rngs.params(), (), param_dtype))

    self._active_dims = (
      active_dims if isinstance(active_dims, list) else slice(active_dims)
    )

  def __call__(self, x1: jax.Array, x2: jax.Array = None):
    if x2 is None:
      x2 = x1

    cov = exponentiated_quadratic(
      x1[..., self._active_dims],
      x2[..., self._active_dims],
      jnp.square(jnp.exp(self.log_sigma)),
      jnp.exp(self.log_rho),
    )
    return cov


# pylint: disable=invalid-name
def exponentiated_quadratic(
  x1: jax.Array,
  x2: jax.Array,
  sigma: float,
  rho: float | jax.Array,
):
  """Exponentiated-quadratic convariance function.

  Args:
    x1:   (`n x p`)-dimensional set of data points
    x2: (`m x p`)-dimensional set of data points
    sigma: the standard deviation of the kernel function
    rho: the length-scale of the kernel function. Can be a float or a
      :math:`p`-dimensional vector if ARD-behaviour is desired

  Returns:
    returns a (`n x m`)-dimensional kernel matrix
  """

  def _exponentiated_quadratic(x, y, sigma, rho):
    x_e = jnp.expand_dims(x, 1) / rho
    y_e = jnp.expand_dims(y, 0) / rho
    d = jnp.sum(jnp.square(x_e - y_e), axis=2)
    K = sigma * jnp.exp(-0.5 * d)
    return K

  return _exponentiated_quadratic(x1, x2, sigma, rho)


# pylint: disable=invalid-name
def periodic(x1: jax.Array, x2: jax.Array, period, sigma, rho):
  """Periodic convariance function.

  Args:
    x1: (`n x p`)-dimensional set of data points
    x2: (`m x p`)-dimensional set of data points
    period: the period
    sigma: the standard deviation of the kernel function
    rho: the length-scale of the kernel function. Can be a float or a
      :math:`p`-dimensional vector if ARD-behaviour is desired

  Returns:
    returns a (`n x m`)-dimensional Gram matrix
  """

  def _periodic(x, y, period, sigma, rho):
    x_e = jnp.expand_dims(x, 1)
    y_e = jnp.expand_dims(y, 0)
    r2 = jnp.sum((x_e - y_e) ** 2, axis=2)
    r = jnp.sqrt(r2)
    K = sigma * jnp.exp(-2 / rho**2 * jnp.sin(jnp.pi * r / period) ** 2)
    return K

  return _periodic(x1, x2, period, sigma, rho)
