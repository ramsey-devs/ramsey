import flax
import jax
import numpyro.distributions
import numpyro.distributions as dist
from chex import assert_axis_dimension, assert_rank
from flax import nnx
from flax.nnx import rnglib
from flax.nnx.module import first_from
from jax import numpy as jnp
from numpyro.distributions import kl_divergence

from ramsey._src.family import Family, Gaussian

__all__ = ["NP"]


class NP(nnx.Module):
  """A neural process.

  Implements the core structure of a vanilla (latent) neural process
  :cite:p:`garnelo18conditional,garnelo2018neural`.

  Args:
    decoder: the decoder can be any network, but is typically an MLP. Note
      that the _last_ layer of the decoder needs to
      have twice the number of nodes as the data you try to model
    latent_encoder: the latent encoder
      can be any network, but is typically an MLP. The first element of
      the tuple is a neural network used before the aggregation step,
      while the second element of the tuple encodes is a neural network
      used to compute mean(s) and standard deviation(s) of the latent
      Gaussian.
    deterministic_encoder: the deterministic encoder can be any network,
      but is typically an MLP
    family: distributional family of the response variable
  """

  def __init__(
    self,
    decoder: nnx.Module,
    deterministic_encoder: flax.nnx.Module | None = None,
    latent_encoder: tuple[flax.nnx.Module, flax.nnx.Module] | None = None,
    family: Family = Gaussian(),
    *,
    rngs: rnglib.Rngs | None = None,
  ):
    self.rngs = rngs
    self._decoder = decoder
    self._family = family
    self._latent_encoder = latent_encoder
    self._deterministic_encoder = deterministic_encoder
    if latent_encoder is None and deterministic_encoder is None:
      raise ValueError("either latent or deterministic encoder needs to be set")
    if latent_encoder is not None:
      self._latent_encoder, self._latent_variable_encoder = (
        latent_encoder[0],
        latent_encoder[1],
      )

  def __call__(
    self,
    x_context: jax.Array,
    y_context: jax.Array,
    x_target: jax.Array,
    *,
    rngs: rnglib.Rngs | None = None,
  ) -> numpyro.distributions.Distribution:
    """Transform the inputs through the neural process.

    Args:
      x_context: input data of dimension
        (*batch_dims, spatial_dims..., feature_dims)
      y_context: input data of dimension
        (*batch_dims, spatial_dims..., response_dims)
      x_target: input data of dimension
        (*batch_dims, spatial_dims..., feature_dims)
      rngs: a rnglib.Rngs object for random seeds

    Returns:
        returns the predictive distribution of y_target
    """
    assert_rank([x_context, y_context, x_target], 3)
    _, num_observations, _ = x_target.shape

    if self._latent_encoder is not None:
      rngs = first_from(
        rngs, self.rngs, error_msg="no 'rngs' argument provided"
      )
      rng = rngs["sample"]()
      z_latent = self._encode_latent(x_context, y_context).sample(rng)
    else:
      z_latent = None

    z_deterministic = self._encode_deterministic(x_context, y_context, x_target)
    representation = self._concat_and_tile(
      z_deterministic, z_latent, num_observations
    )
    pred_fn = self._decode(representation, x_target, y_context)

    return pred_fn

  def loss(
    self,
    x_context: jax.Array,
    y_context: jax.Array,
    x_target: jax.Array,
    y_target: jax.Array,
    *,
    rngs: rnglib.Rngs | None = None,
  ) -> jax.Array:
    """Transform the inputs through the neural process.

    Args:
      x_context: input data of dimension
        (*batch_dims, spatial_dims..., feature_dims)
      y_context: input data of dimension
        (*batch_dims, spatial_dims..., response_dims)
      x_target: input data of dimension
        (*batch_dims, spatial_dims..., feature_dims)
      y_target: input data of dimension
        (*batch_dims, spatial_dims..., response_dims)
      rngs: a rnglib.Rngs object for random seeds

    Returns:
      returns the negative ELBO
    """
    _, num_observations, _ = x_target.shape

    if self._latent_encoder is not None:
      rngs = first_from(
        rngs, self.rngs, error_msg="no 'rngs' argument provided"
      )
      rng = rngs["sample"]()
      prior = self._encode_latent(x_context, y_context)
      posterior = self._encode_latent(x_target, y_target)
      z_latent = posterior.sample(rng)
      kl = jnp.sum(kl_divergence(posterior, prior), axis=-1)
    else:
      z_latent = None
      kl = 0

    z_deterministic = self._encode_deterministic(x_context, y_context, x_target)
    representation = self._concat_and_tile(
      z_deterministic, z_latent, num_observations
    )
    pred_fn = self._decode(representation, x_target, y_target)
    loglik = jnp.sum(pred_fn.log_prob(y_target), axis=1)
    elbo = jnp.mean(loglik - kl)

    return -elbo

  @staticmethod
  def _concat_and_tile(z_deterministic, z_latent, num_observations):
    if z_deterministic is None:
      representation = z_latent
    elif z_latent is None:
      representation = z_deterministic
    else:
      representation = jnp.concatenate([z_deterministic, z_latent], axis=-1)
    assert_axis_dimension(representation, 1, 1)
    representation = jnp.tile(representation, [1, num_observations, 1])
    assert_axis_dimension(representation, 1, num_observations)
    return representation

  def _encode_deterministic(
    self,
    x_context: jax.Array,
    y_context: jax.Array,
    x_target: jax.Array,
  ):
    if self._deterministic_encoder is None:
      return None
    xy_context = jnp.concatenate([x_context, y_context], axis=-1)
    z_deterministic = self._deterministic_encoder(xy_context)
    z_deterministic = jnp.mean(z_deterministic, axis=1, keepdims=True)
    return z_deterministic

  def _encode_latent(self, x_context: jax.Array, y_context: jax.Array):
    xy_context = jnp.concatenate([x_context, y_context], axis=-1)
    z_latent = self._latent_encoder(xy_context)  # type: ignore[operator,misc]
    return self._encode_latent_gaussian(z_latent)

  def _encode_latent_gaussian(self, z_latent):
    z_latent = jnp.mean(z_latent, axis=1, keepdims=True)
    z_latent = self._latent_variable_encoder(z_latent)
    mean, sigma = jnp.split(z_latent, 2, axis=-1)
    sigma = 0.1 + 0.9 * jax.nn.sigmoid(sigma)
    return dist.Normal(loc=mean, scale=sigma)

  def _decode(
    self, representation: jax.Array, x_target: jax.Array, y: jax.Array
  ):
    target = jnp.concatenate([representation, x_target], axis=-1)
    target = self._decoder(target)
    family = self._family(target)
    self._check_posterior_predictive_axis(family, x_target, y)
    return family

  @staticmethod
  def _check_posterior_predictive_axis(
    family: dist.Distribution,
    x_target: jax.Array,
    y: jax.Array,  # pylint: disable=invalid-name
  ):
    assert_axis_dimension(family.mean, 0, x_target.shape[0])
    assert_axis_dimension(family.mean, 1, x_target.shape[1])
    assert_axis_dimension(family.mean, 2, y.shape[2])
