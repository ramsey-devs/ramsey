import flax
import jax
from jax import numpy as jnp

from ramsey._src.family import Family, Gaussian
from ramsey._src.neural_process.attentive_neural_process import ANP

__all__ = ["DANP"]

from ramsey._src.nn.attention.attention import Attention


# ruff: noqa: PLR0913
class DANP(ANP):
  """A doubly-attentive neural process.

  Implements the core structure of a 'doubly-attentive' neural process
  :cite:p:`kim2018attentive`.

  Args:
    decoder: the decoder can be any network, but is typically an MLP. Note
      that the _last_ layer of the decoder needs to
      have twice the number of nodes as the data you try to model
    latent_encoder: a tuple of two `flax.linen.Module`s and an attention object.
      The first and last elements are the usual modules required for a
      neural process, the attention object computes self-attention before the
       aggregation
    deterministic_encoder: a tuple of a `flax.linen.Module` and an Attention
      object. The first `attention` object is used for self-attention,
      the second one is used for cross-attention
    family: distributional family of the response variable
  """

  decoder: flax.linen.Module
  deterministic_encoder: (
    tuple[flax.linen.Module, Attention, Attention] | None
  ) = None  # type: ignore[assignment]
  latent_encoder: (
    tuple[flax.linen.Module, Attention, flax.linen.Module] | None
  ) = None  # type: ignore[assignment]
  family: Family = Gaussian()

  def setup(self):
    """Construct all networks."""
    if self.latent_encoder is None and self.deterministic_encoder is None:
      raise ValueError("either latent or deterministic encoder needs to be set")
    if self.latent_encoder is not None:
      (
        self._latent_encoder,
        self._latent_self_attention,
        self._latent_variable_encoder,
      ) = self.latent_encoder
    if self.deterministic_encoder is not None:
      (
        self._deterministic_encoder,
        self._deterministic_self_attention,
        self._deterministic_cross_attention,
      ) = self.deterministic_encoder
    self._decoder = self.decoder
    self._family = self.family

  def _encode_latent(self, x_context: jax.Array, y_context: jax.Array):
    xy_context = jnp.concatenate([x_context, y_context], axis=-1)
    z_latent = self._latent_encoder(xy_context)  # type: ignore[operator, misc]
    z_latent = self._latent_self_attention(z_latent, z_latent, z_latent)
    return self._encode_latent_gaussian(z_latent)

  def _encode_deterministic(
    self,
    x_context: jax.Array,
    y_context: jax.Array,
    x_target: jax.Array,
  ):
    xy_context = jnp.concatenate([x_context, y_context], axis=-1)
    z_deterministic = self._deterministic_encoder(xy_context)  # type: ignore[misc]
    z_deterministic = self._deterministic_self_attention(
      z_deterministic, z_deterministic, z_deterministic
    )
    z_deterministic = self._deterministic_cross_attention(
      x_context, z_deterministic, x_target
    )
    return z_deterministic
