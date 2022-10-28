from typing import Tuple

import numpyro.distributions as dist
from jax import numpy as jnp
from jax import random

from ramsey._src.datasets import (
    M4_DAILY,
    M4_HOURLY,
    M4_MONTHLY,
    M4_QUARTERLY,
    M4_WEEKLY,
    M4_YEARLY,
    M4Dataset,
)
from ramsey.covariance_functions import exponentiated_quadratic

m4_datasets = {
    "m4_hourly": {
        "key": M4_HOURLY,
        "n_observations": 700,
        "n_forecasts": 48,
        "series_prefix": "H",
    },
    "m4_daily": {
        "key": M4_DAILY,
        "n_observations": 93,
        "n_forecasts": 14,
        "series_prefix": "D",
    },
    "m4_weekly": {
        "key": M4_WEEKLY,
        "n_observations": 80,
        "n_forecasts": 13,
        "series_prefix": "W",
    },
    "m4_monthly": {
        "key": M4_MONTHLY,
        "n_observations": 42,
        "n_forecasts": 18,
        "series_prefix": "M",
    },
    "m4_quarterly": {
        "key": M4_QUARTERLY,
        "n_observations": 16,
        "n_forecasts": 8,
        "series_prefix": "Q",
    },
    "m4_yearly": {
        "key": M4_YEARLY,
        "n_observations": 13,
        "n_forecasts": 6,
        "series_prefix": "Y",
    },
}


def load_m4_dataset(dset_name: str) -> Tuple[dict, dict]:
    dset = m4_datasets[dset_name]["key"]
    train, test = M4Dataset.load(dset)
    return train, test


def get_m4_time_series(
    name: str, train: dict, test: dict
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    y_train = jnp.array(train[name])
    t_train = jnp.arange(start=0, stop=len(y_train), step=1.0, dtype=float)

    y_test = jnp.array(test[name])
    t_test = jnp.arange(
        start=len(y_train),
        stop=len(y_train) + len(y_test),
        step=1.0,
        dtype=float,
    )

    t_train = jnp.reshape(t_train, (len(t_train), 1))
    y_train = jnp.reshape(y_train, (len(y_train), 1))

    t_test = jnp.reshape(t_test, (len(t_test), 1))
    y_test = jnp.reshape(y_test, (len(y_test), 1))

    return t_train, y_train, t_test, y_test


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
