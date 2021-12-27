import numpyro.distributions as dist
from jax import numpy as np
from jax import random

from ramsey.covariance_functions import exponentiated_quadratic


# pylint: disable=too-many-locals,invalid-name
def sample_from_sinus_function(key, batch_size=10, num_observations=100):
    x = np.linspace(-np.pi, np.pi, num_observations).reshape(
        (num_observations, 1)
    )
    ys = []
    fs = []
    for _ in range(batch_size):
        key, sample_key1, sample_key2, sample_key3 = random.split(key, 4)
        a = 2 * random.uniform(sample_key1) - 1
        b = random.uniform(sample_key2) - 0.5
        f = a * np.sin(x - b)
        y = f + random.normal(sample_key3, shape=(num_observations, 1)) * 0.10
        fs.append(f.reshape((1, num_observations, 1)))
        ys.append(y.reshape((1, num_observations, 1)))

    x = np.tile(x, [batch_size, 1, 1])
    y = np.vstack(np.array(ys))
    f = np.vstack(np.array(fs))

    return (x, y), f


# pylint: disable=too-many-locals,invalid-name
def sample_from_gaussian_process(
    key, batch_size=10, num_observations=100, num_dim=1
):
    x = random.normal(key, shape=(num_observations * num_dim,)).reshape(
        (num_observations, num_dim)
    )
    ys = []
    fs = []
    for _ in range(batch_size):
        key, sample_key1, sample_key2, sample_key3, sample_key4 = random.split(
            key, 5
        )
        rho = dist.InverseGamma(1, 1).sample(sample_key1)
        sigma = dist.InverseGamma(5, 5).sample(sample_key2)
        K = exponentiated_quadratic(x, x, rho, sigma)

        f = random.multivariate_normal(
            sample_key3,
            mean=np.zeros(num_observations),
            cov=K + np.diag(np.ones(num_observations)) * 1e-5,
        )
        y = random.multivariate_normal(
            sample_key4, mean=f, cov=np.eye(num_observations) * 0.05
        )
        fs.append(f.reshape((1, num_observations, 1)))
        ys.append(y.reshape((1, num_observations, 1)))

    x = np.tile(x, [batch_size, 1, 1])
    y = np.vstack(np.array(ys))
    f = np.vstack(np.array(fs))

    return (x, y), f
