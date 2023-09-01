from collections import namedtuple
from typing import NamedTuple

from numpyro import distributions as dist

import pandas as pd
from jax import numpy as jnp
from jax import random as jr

from ramsey._src.datasets import M4Dataset
from ramsey.kernels import exponentiated_quadratic


# pylint: disable=too-many-locals,invalid-name
def load_m4_time_series_data(
    interval: str = "hourly",
    drop_na: bool = True
) -> NamedTuple:
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
    NamedTuple
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
    return namedtuple("data", "y x train_idxs test_idxs")(y, x, train_idxs, test_idxs)


# pylint: disable=too-many-locals,invalid-name
def sample_from_polynomial_function(
    seed, batch_size=10, order=1, num_observations=100, sigma=0.1
):
    x = jnp.linspace(-jnp.pi, jnp.pi, num_observations).reshape(
        (num_observations, 1)
    )
    ys = []
    fs = []
    for _ in range(batch_size):
        y_rng_key, coeff_rng_key, seed = jr.split(seed, 3)
        coeffs = list(jr.uniform(coeff_rng_key, shape=(order + 1, 1)) - 1)
        f = []
        for i in range(order + 1):
            f += coeffs[i] * x**i

        y = f + jr.normal(y_rng_key, shape=(num_observations, 1)) * sigma
        fs.append(f.reshape((1, num_observations, 1)))
        ys.append(y.reshape((1, num_observations, 1)))

    x = jnp.tile(x, [batch_size, 1, 1])
    y = jnp.vstack(jnp.array(ys))
    f = jnp.vstack(jnp.array(fs))

    return namedtuple("data", "y x f")(y, x, f)


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

    return namedtuple("data", "y x f")(y, x, f)


# pylint: disable=too-many-locals,invalid-name
def sample_from_gaussian_process(
    seed, batch_size=10, num_observations=100, num_dim=1, rho=None, sigma=None
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

    return namedtuple("data", "y x f")(y, x, f)

