"""
Gaussian process regression
===========================

This example implements the training and prediction of a Gaussian process
regression model.

References
----------

[1] Williams, Christopher KI, and Carl Edward Rasmussen. "Gaussian Processes for
    Machine Learning." MIT press, 2006.
"""

import haiku as hk

from jax import numpy as jnp, random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from ramsey.train import train_gaussian_process
from ramsey.data import sample_from_gaussian_process
from ramsey.covariance_functions import ExponentiatedQuadratic
from ramsey.models import GP


def data(key, rho, sigma, n=1000):
    (x_target, y_target), f_target = sample_from_gaussian_process(
        key, batch_size=1, num_observations=n, rho=rho, sigma=sigma
    )
    return (x_target.reshape(n, 1), y_target.reshape(n, 1)), f_target.reshape(n, 1)


def _gaussian_process(**kwargs):
    gp = GP(ExponentiatedQuadratic())
    return gp(**kwargs)


def train_gp(key, x, y):
    _, init_key, train_key = random.split(key, 3)
    gaussian_process = hk.transform(_gaussian_process)
    params = gaussian_process.init(init_key, x=x)

    params, _ = train_gaussian_process(
      gaussian_process,
      params,
      train_key,
      x=x,
      y=y,
    )

    return gaussian_process, params


def plot(key, gaussian_process, params, x, y, f, train_idxs):
    key, sample_key = random.split(key, 2)

    _, ax = plt.subplots(figsize=(8, 3))
    srt_idxs = jnp.argsort(jnp.squeeze(x))
    ax.plot(
        jnp.squeeze(x)[srt_idxs], jnp.squeeze(f)[srt_idxs], color="black", alpha=0.5
    )
    ax.scatter(
        jnp.squeeze(x[train_idxs, :]), jnp.squeeze(y[train_idxs, :]),
        color="red", marker="+", alpha=0.5)

    key, apply_key = random.split(key, 2)
    y_star = gaussian_process.apply(
        params=params,
        rng=apply_key,
        x=x[train_idxs, :],
        y=y[train_idxs, :],
        x_star=x,
    ).mean()
    ax.plot(jnp.squeeze(x)[srt_idxs], jnp.squeeze(y_star)[srt_idxs], color="blue")
    lgd = ax.legend(
        handles=[mpatches.Patch(color="black", alpha=0.5, label='Latent function ' + r"$f \sim GP$"),
                 mpatches.Patch(color="red", alpha=0.45, label='Training data'),
                 mpatches.Patch(color="blue", alpha=0.45, label='Posterior mean'),
                 ],
        loc='best',
        frameon=False
    )
    plt.show()


def run():
    rng_seq = hk.PRNGSequence(123)
    n_train = 30

    (x, y), f = data(next(rng_seq), 0.25, 3.0)
    train_idxs = random.choice(
      next(rng_seq), jnp.arange(x.shape[0]), shape=(n_train,), replace=False
    )

    x_train, y_train = x[train_idxs, :], y[train_idxs, :]
    gaussian_process, params = train_gp(next(rng_seq),  x_train, y_train)

    plot(
        next(rng_seq), gaussian_process, params,
        x, y, f, train_idxs,
    )


if __name__ == "__main__":
    run()
