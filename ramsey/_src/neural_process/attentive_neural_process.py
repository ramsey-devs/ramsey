import flax
from chex import assert_axis_dimension
from flax import nnx
from flax.nnx import rnglib
from jax import numpy as jnp

from ramsey._src.family import Family, Gaussian
from ramsey._src.neural_process.neural_process import NP
from ramsey._src.nn.attention.attention import Attention

__all__ = ["ANP"]


# ruff: noqa: PLR0913
class ANP(NP):
  """An attentive neural process.

  Implements the core structure of an attentive neural process
  :cite:p:`kim2018attentive`.

  Args:
    decoder: the decoder can be any network, but is typically an MLP. Note
      that the _last_ layer of the decoder needs to
      have twice the number of nodes as the data you try to model
    deterministic_encoder: a tuple of a `flax.linen.Module` and an Attention obj
    latent_encoder: an optional tuple of two `flax.linen.Module`s. The latent
      encoder can be any network, but is typically an MLP. The first element of
      the tuple is a neural network used before the aggregation step, while the
      second   element of the tuple encodes is a neural network used to
      compute mean(s) and standard deviation(s) of the latent Gaussian.
    family: distributional family of the response variable
  """

  def __init__(
    self,
    decoder: nnx.Module,
    deterministic_encoder: tuple[flax.nnx.Module, Attention] | None = None,
    latent_encoder: tuple[flax.nnx.Module, flax.nnx.Module] | None = None,
    family: Family = Gaussian(),
    *,
    rngs: rnglib.Rngs | None = None,
  ):
    super().__init__(
      decoder, deterministic_encoder, latent_encoder, family, rngs=rngs
    )
    if latent_encoder is not None:
      self._latent_encoder, self._latent_variable_encoder = (
        latent_encoder[0],
        latent_encoder[1],
      )
    if deterministic_encoder is not None:
      self._deterministic_encoder = deterministic_encoder[0]
      self._deterministic_cross_attention = deterministic_encoder[1]

  @staticmethod
  def _concat_and_tile(z_deterministic, z_latent, num_observations):
    if z_latent is not None:
      if z_latent.shape[1] == 1:
        z_latent = jnp.tile(z_latent, [1, num_observations, 1])
      representation = jnp.concatenate([z_deterministic, z_latent], axis=-1)
    else:
      representation = z_deterministic
    assert_axis_dimension(representation, 1, num_observations)
    return representation

  def _encode_deterministic(self, x_context, y_context, x_target):
    xy_context = jnp.concatenate([x_context, y_context], axis=-1)
    z_deterministic = self._deterministic_encoder(xy_context)
    z_deterministic = self._deterministic_cross_attention(
      x_context, z_deterministic, x_target
    )
    return z_deterministic
