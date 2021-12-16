import haiku as hk
import jax
import jax.numpy as np
import jax.random as random
import numpyro.distributions as dist

from pax.gaussian_process.covariance.base import covariance
from pax.gaussian_process.covariance.stationary import exponentiated_quadratic
from pax.neural_process import NP
from pax.training.train import train

import matplotlib.pyplot as plt

batch_size = 10
n = 50
p = 1

key = random.PRNGKey(0)
x = random.normal(random.PRNGKey(0), shape=(n * p,)).reshape((n, p))
ys = []
for i in range(batch_size):
    key, sample_key1, sample_key2, sample_key3 = random.split(key, 4)
    rho = dist.InverseGamma(5, 5).sample(sample_key1)
    sigma = dist.InverseGamma(5, 5).sample(sample_key2)
    K = covariance(
        exponentiated_quadratic,
        {"rho": rho, "sigma": sigma},
        x, x
    )
    y = jax.random.multivariate_normal(
        sample_key3,
        mean=np.zeros(n),
        cov=K + np.diag(np.ones(n)) * 0.05
    ).reshape((1, n, 1))
    ys.append(y)

x_target = np.tile(x, [batch_size, 1, 1])
y_target = np.vstack(np.array(ys))


n_context = int(np.floor(n / 4))
idxs_context = random.choice(
    random.PRNGKey(0),
    np.arange(n), shape=(n_context,), replace=False
)

x_context = x_target[:, idxs_context, :]
y_context = y_target[:, idxs_context, :]


# for i in range(batch_size):
#     x = np.squeeze(x_target[i, :, :])
#     y = np.squeeze(y_target[i, :, :])
#     idxs = np.argsort(x)
#     plt.scatter(x[idxs], y[idxs], marker="+")
# plt.show()


def deterministic_encoder_fn(x):
    mlp = hk.Sequential([
        hk.Linear(10),
        jax.nn.relu,
        hk.Linear(10),
        jax.nn.relu,
        hk.Linear(2)
    ])
    return mlp(x)


def latent_encoder_fn(x):
    mlp = hk.Sequential([
        hk.Linear(10),
        jax.nn.relu,
        hk.Linear(10),
        jax.nn.relu,
        hk.Linear(10)
    ])
    return mlp(x)


def decoder_fn(x):
    mlp = hk.Sequential([
        hk.Linear(10),
        jax.nn.relu,
        hk.Linear(2)
    ])
    return mlp(x)


def f(x_context, y_context, x_target, **kwargs):
    np = NP(deterministic_encoder_fn, latent_encoder_fn, 3, decoder_fn)
    return np(x_context, y_context, x_target, **kwargs)


f = hk.transform(f)

key, init_key = random.split(key)
params = f.init(
    init_key,
    x_context=x_context,
    y_context=y_context,
    x_target=x_target
)

key, apply_key = random.split(key)
y_star = f.apply(
    rng=apply_key,
    params=params,
    x_context=x_context,
    y_context=y_context,
    x_target=x_target
)

key, train_key = random.split(key)
train(f, params, train_key,
      x_context=x_context, y_context=y_context, x_target=x_target, y_target=y_target
      )
