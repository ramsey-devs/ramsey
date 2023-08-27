from typing import Tuple


from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

import pandas as pd
from chex import Array
from jax import nn
from jax import numpy as jnp
from jax import random as jr

from ramsey._src.datasets import M4Dataset
from ramsey.kernels import exponentiated_quadratic


# pylint: disable=too-many-locals,invalid-name
def load_m4_time_series_data(
    interval: str = "hourly", drop_na=True
) -> Tuple[Tuple[Array, Array], Tuple[Array, Array]]:
    """
    Load an M4 data set

    Parameters
    ----------
    interval: str
        either of "hourly", "daily", "weekly", "monthly", "yearly"
    drop_na: bool
        drop rows that contain NA values

    Returns
    -------
    Tuple[Tuple[Array, Array], Tuple[Array, Array]]
        a tuple of tuples. The first tuple consists of two JAX arrays
        where the first element are the time series observations (Y)
        and the second are features (X). The second tuple are arrays of
        training and testing indexes that can be used to subset Y and X
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
    return (y, x), (train_idxs, test_idxs)


# pylint: disable=too-many-locals,invalid-name
def sample_from_polynomial_function(
    rng, batch_size=10, order=1, num_observations=100, sigma=0.1
):
    x = jnp.linspace(-jnp.pi, jnp.pi, num_observations).reshape(
        (num_observations, 1)
    )
    ys = []
    fs = []
    for _ in range(batch_size):
        coeffs = list(jr.uniform(next(rng), shape=(order + 1, 1)) - 1)
        f = 0
        for i in range(order + 1):
            f += coeffs[i] * x**i

        y = f + jr.normal(next(rng), shape=(num_observations, 1)) * sigma

        fs.append(f.reshape((1, num_observations, 1)))
        ys.append(y.reshape((1, num_observations, 1)))

    x = jnp.tile(x, [batch_size, 1, 1])
    y = jnp.vstack(jnp.array(ys))
    f = jnp.vstack(jnp.array(fs))

    return (x, y), f


# pylint: disable=too-many-locals,invalid-name
def sample_from_sine_function(seed, batch_size=10, num_observations=100):
    x = jnp.linspace(-jnp.pi, jnp.pi, num_observations).reshape(
        (num_observations, 1)
    )
    ys = []
    fs = []
    for _ in range(batch_size):
        sample_key1, sample_key2, sample_key3, seed = jr.split(seed, 4)
        a = 2 * jr.uniform(sample_key1) - 1
        b = jr.uniform(sample_key2) - 0.5
        f = a * jnp.sin(x - b)
        y = f + jr.normal(sample_key3, shape=(num_observations, 1)) * 0.10
        fs.append(f.reshape((1, num_observations, 1)))
        ys.append(y.reshape((1, num_observations, 1)))

    x = jnp.tile(x, [batch_size, 1, 1])
    y = jnp.vstack(jnp.array(ys))
    f = jnp.vstack(jnp.array(fs))

    return (x, y), f


# pylint: disable=too-many-locals,invalid-name
def sample_from_gaussian_process(
    key, batch_size=10, num_observations=100, num_dim=1, rho=None, sigma=None
):
    x = jnp.linspace(-jnp.pi, jnp.pi, num_observations).reshape(
        (num_observations, num_dim)
    )
    ys = []
    fs = []
    for _ in range(batch_size):
        sample_key1, sample_key2, sample_key3, sample_key4, seed = jr.split(
            seed, 5
        )
        if rho is None:
            rho = tfd.InverseGamma(1, 1).sample(sample_key1)
        if sigma is None:
            sigma = tfd.InverseGamma(5, 5).sample(sample_key2)
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

    return (x, y), f


# pylint: disable=too-many-locals,invalid-name
def sample_from_negative_binomial_linear_model(
    seed, batch_size=10, num_observations=100, num_dim=1
):
    x_key, seed = jr.split(seed)
    x = jr.normal(x_key, (num_observations, num_dim))
    ys, fs = [], []
    for _ in range(batch_size):
        alpha_key, beta_key, seed = jr.split(seed, 3)
        alpha = tfd.Normal(1.0, 3.0).sample(alpha_key)
        beta = tfd.Normal(10.0, 3.0).sample(beta_key, (num_dim,))
        f = nn.softplus(alpha + x @ beta)
        y_key, seed = jr.split(seed)
        y = tfd.Poisson(f).sample(y_key)
        fs.append(f.reshape((1, num_observations, 1)))
        ys.append(y.reshape((1, num_observations, 1)))

    x = jnp.tile(x, [batch_size, 1, 1])
    y = jnp.vstack(jnp.array(ys))
    f = jnp.vstack(jnp.array(fs))

    return (x, y), f


# pylint: disable=too-many-locals,invalid-name
def sample_from_linear_model(
    seed, batch_size=10, num_observations=100, num_dim=1, noise_scale=None
):
    x_key, seed = jr.split(seed)
    x = jr.normal(x_key, (num_observations, num_dim))
    ys = []
    fs = []
    for _ in range(batch_size):
        beta_key, seed = jr.split(seed)
        beta = tfd.Normal(0.0, 2.0).sample(beta_key, (num_dim,))
        if noise_scale is None:
            noise_key, seed = jr.split(seed)
            noise_scale = tfd.Gamma(1.0, 10.0).sample(noise_key)
        f = x @ beta
        y_key, seed = jr.split(seed)
        y = f + jr.normal(y_key, f.shape) * noise_scale
        fs.append(f.reshape((1, num_observations, 1)))
        ys.append(y.reshape((1, num_observations, 1)))

    x = jnp.tile(x, [batch_size, 1, 1])
    y = jnp.vstack(jnp.array(ys))
    f = jnp.vstack(jnp.array(fs))

    return (x, y), f
