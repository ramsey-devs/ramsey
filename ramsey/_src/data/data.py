from collections import namedtuple

import jax
import pandas as pd
from jax import numpy as jnp
from jax import random as jr
from numpyro import distributions as dist

from ramsey._src.data.dataset_m4 import M4Dataset
from ramsey._src.experimental.kernel.stationary import exponentiated_quadratic


# pylint: disable=too-many-locals,invalid-name
def m4_data(interval: str = "hourly", drop_na: bool = True):
  """Load a data set from the M4 competition.

  Args:
    interval: either of "hourly", "daily", "weekly", "monthly", "yearly"
    drop_na: drop rows that contain NA values

  Returns:
    returns a named tuple with outputs (y), inputs (x), and training and
    testing indexes for the input-output paris
  """
  train, test = M4Dataset().load(interval)
  df = pd.concat([train, test.reindex(train.index)], axis=1)
  if drop_na:
    df = df.dropna()
  y = df.values
  y = y.reshape((*y.shape, 1))
  x = jnp.arange(y.shape[1]) / train.shape[1]
  x = jnp.tile(x, [y.shape[0], 1]).reshape((y.shape[0], y.shape[1], 1))
  train_idxs = jnp.arange(train.shape[1])
  test_idxs = jnp.arange(test.shape[1]) + train.shape[1]

  return namedtuple("data", ["y", "x", "train_idxs", "test_idxs"])(  # type: ignore
    y, x, train_idxs, test_idxs
  )


# pylint: disable=too-many-locals,invalid-name
def sample_from_sine_function(
  rng_key: jax.Array, batch_size: int = 10, num_observations: int = 100
):
  r"""Sample from a noisy sine function.

  Creates samples from a noisy sine functions. For each batch,
  chooses a new hyper-parameters configuration.

  The inputs, `x` of the sine function have dimensionality
  :math:`b \times n \times 1`, where `b` is the batch size and `n` is the
  number of observations per batch. The outputs and latent functions
  realizations have dimension :math:`b \times n \times 1` as well.


  Args:
    rng_key: a JAX random key for seeding
    batch_size: size of batch
    num_observations: number of observations per batch
    rho: the lengthscale of the kernel function
    sigma: the standard deviation of the kernel function

  Returns:
    a tuple consisting of outputs (y), inputs (x) and latent GP
      realization (f) where
  """
  x = jnp.linspace(-jnp.pi, jnp.pi, num_observations).reshape(
    (num_observations, 1)
  )
  ys = []
  fs = []
  for _ in range(batch_size):
    sample_key1, sample_key2, sample_key3, rng_key = jr.split(rng_key, 4)
    a = 2 * jr.uniform(sample_key1) - 1
    b = jr.uniform(sample_key2) - 0.5
    f = a * jnp.sin(x - b)
    y = f + jr.normal(sample_key3, shape=(num_observations, 1)) * 0.10
    fs.append(f.reshape((1, num_observations, 1)))
    ys.append(y.reshape((1, num_observations, 1)))

  x = jnp.tile(x, [batch_size, 1, 1])
  y = jnp.vstack(jnp.array(ys))
  f = jnp.vstack(jnp.array(fs))

  return namedtuple("data", "y x f")(y, x, f)  # type: ignore[call-arg]


# pylint: disable=too-many-locals,invalid-name
def sample_from_gaussian_process(
  rng_key: jax.Array,
  batch_size: int = 10,
  num_observations: int = 100,
  rho: float | None = None,
  sigma: float | None = None,
):
  r"""Sample from a Gaussian process.

  Creates samples from a Gaussian process with exponentiated quadratic
  covariance function. For each batch, chooses a new hyperparameter
  configuration where `rho`, the kernel lengthscale is drawn
  from an `InverseGamma(1, 1)` and sigma, the kernel lengthscale, is drawn
  from an `InverseGamma(5, 5)`.

  The inputs, `x` of the Gaussian process have dimensionality
  :math:`b \times n \times 1`, where `b` is the batch size and `n` is the
  number of observations per batch. The outputs and latent functions
  realizations have dimension :math:`b \times n \times 1` as well.

  Args:
    rng_key: a random key for seeding
    batch_size: size of batch
    num_observations: number of observations per batch
    rho: the lengthscale of the kernel function
    sigma: the standard deviation of the kernel function

  Returns:
    a tuple consisting of outputs (y), inputs (x) and latent GP
    realization (f) where
  """
  x = jnp.linspace(-jnp.pi, jnp.pi, num_observations).reshape(
    (num_observations, 1)
  )
  ys = []
  fs = []
  for _ in range(batch_size):
    sample_key1, sample_key2, sample_key3, sample_key4, rng_key = jr.split(
      rng_key, 5
    )
    if rho is None:
      rho = dist.InverseGamma(1, 1).sample(sample_key1)
    if sigma is None:
      sigma = dist.InverseGamma(5, 5).sample(sample_key2)
    K = exponentiated_quadratic(x, x, sigma, rho)

    f = jr.multivariate_normal(
      sample_key3,
      mean=jnp.zeros(num_observations),
      cov=K + jnp.diag(jnp.ones(num_observations)) * 1e-5,
    )
    y = jr.multivariate_normal(
      sample_key4, mean=f, cov=jnp.eye(num_observations) * 0.05
    )
    fs.append(f.reshape((1, num_observations, 1)))
    ys.append(y.reshape((1, num_observations, 1)))

  x = jnp.tile(x, [batch_size, 1, 1])
  y = jnp.vstack(jnp.array(ys))
  f = jnp.vstack(jnp.array(fs))

  return namedtuple("data", "y x f")(y, x, f)  # type: ignore[call-arg]
