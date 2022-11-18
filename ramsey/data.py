from typing import Tuple

import numpyro.distributions as dist
import pandas as pd
from chex import Array
from jax import numpy as jnp
from jax import random

from ramsey._src.datasets import M4Dataset
from ramsey.covariance_functions import exponentiated_quadratic


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
def sample_from_sine_function(key, batch_size=10, num_observations=100):
    x = jnp.linspace(-jnp.pi, jnp.pi, num_observations).reshape(
        (num_observations, 1)
    )
    ys = []
    fs = []
    for _ in range(batch_size):
        key, sample_key1, sample_key2, sample_key3 = random.split(key, 4)
        a = 2 * random.uniform(sample_key1) - 1
        b = random.uniform(sample_key2) - 0.5
        f = a * jnp.sin(x - b)
        y = f + random.normal(sample_key3, shape=(num_observations, 1)) * 0.10
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
        key, sample_key1, sample_key2, sample_key3, sample_key4 = random.split(
            key, 5
        )
        if rho is None:
            rho = dist.InverseGamma(1, 1).sample(sample_key1)
        if sigma is None:
            sigma = dist.InverseGamma(5, 5).sample(sample_key2)
        K = exponentiated_quadratic(x, x, sigma, rho)

        f = random.multivariate_normal(
            sample_key3,
            mean=jnp.zeros(num_observations),
            cov=K + jnp.diag(jnp.ones(num_observations)) * 1e-5,
        )
        y = random.multivariate_normal(
            sample_key4, mean=f, cov=jnp.eye(num_observations) * 0.05
        )
        fs.append(f.reshape((1, num_observations, 1)))
        ys.append(y.reshape((1, num_observations, 1)))

    x = jnp.tile(x, [batch_size, 1, 1])
    y = jnp.vstack(jnp.array(ys))
    f = jnp.vstack(jnp.array(fs))

    return (x, y), f
