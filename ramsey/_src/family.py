import abc

import jax
from jax import numpy as jnp
from numpyro import distributions as nd


class Family(abc.ABC):
  """Distributional family."""

  @abc.abstractmethod
  def __call__(self, target: jnp.ndarray, **kwargs):
    """Compose a NumPyro distribution."""


class Gaussian(Family):
  """Family of Gaussian distributions."""

  def __call__(
    self, target: jax.Array, log_scale: jax.Array | None = None, **kwargs
  ) -> nd.Distribution:
    """Compose a NumPyro distribution."""
    if log_scale is not None:
      mean = target
      scale = jnp.exp(log_scale)
    else:
      mean, log_scale = jnp.split(target, 2, axis=-1)
      scale = 0.1 + 0.9 * jax.nn.softplus(log_scale)
    return nd.Normal(loc=mean, scale=scale)


class NegativeBinomial(Family):
  """Family of negative binomial distributions."""

  def __call__(
    self,
    target: jax.Array,
    log_concentration: jax.Array | None = None,
    **kwargs,
  ) -> nd.Distribution:
    if log_concentration is None:
      mean, log_concentration = jnp.split(target, 2, axis=-1)
    else:
      mean = target
    mean = jnp.exp(mean)
    concentration = jnp.exp(log_concentration)
    return nd.NegativeBinomial2(mean=mean, concentration=concentration)
