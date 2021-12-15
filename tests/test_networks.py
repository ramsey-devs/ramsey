import time
import timeit

import haiku as hk
import jax
import jax.numpy as np
import jax.random as random

from pax.attention import uniform_attention
from pax.gaussian_process.covariance.base import covariance
from pax.gaussian_process.covariance.stationary import rbf, \
    exponentiated_quadratic
import matplotlib.pyplot as plt

from pax.neural_process import NP

n = 20
p = 1

x = random.normal(random.PRNGKey(20), shape=(n * p,)).reshape((n, p))

K = covariance(
    exponentiated_quadratic,
    {"rho": 1.0, "sigma": 1.0},
    x, x
)

y = jax.random.multivariate_normal(
    random.PRNGKey(9),
    mean=np.zeros(n),
    cov=K + np.diag(np.ones(n)) * 0.01
).reshape((n, 1))


def deterministic_encoder_fn(x):
    mlp = hk.Sequential([
        hk.Linear(3),
        jax.nn.relu,
        hk.Linear(10),
        jax.nn.relu,
        hk.Linear(2)
    ])
    return mlp(x)


def latent_encoder_fn(x):
    mlp = hk.Sequential([
        hk.Linear(5), jax.nn.relu,
        hk.Linear(2)
    ])
    return mlp(x)


def decoder_fn(x):
    mlp = hk.Sequential([
        hk.Linear(4), jax.nn.relu
    ])
    return mlp(x)


def f(x, y, z):
    np = NP(deterministic_encoder_fn, latent_encoder_fn, 3, decoder_fn)
    return np(x, y, z)


x = x[np.newaxis, :, :]
x_t = np.hstack([x, x])
y = y[np.newaxis, :, :]

f = hk.transform(f)
params = f.init(random.PRNGKey(1), x=x, y=x, z=x_t)

print(":----------:")
v = f.apply(params=params, x=x, y=y, z=x_t, rng=random.PRNGKey(2))

print(v.shape)
