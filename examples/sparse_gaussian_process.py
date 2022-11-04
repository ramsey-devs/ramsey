"""
Sparse Gaussian process regression
===========================

This example implements the training and prediction of a sparse Gaussian process
regression model.

References
----------

[1] Titsias, Michalis
    "Variational Learning of Inducing Variables in Sparse Gaussian Processes", April 2009
    http://proceedings.mlr.press/v5/titsias09a/titsias09a.pdf
"""

import haiku as hk

from jax import numpy as jnp, random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from ramsey.train import train_sparse_gaussian_process
from ramsey.data import sample_from_gaussian_process
from ramsey.covariance_functions import ExponentiatedQuadratic
from ramsey.models import SparseGP

from jax.config import config
config.update("jax_enable_x64", True)

def data(key, rho, sigma, n=1000):
    (x_target, y_target), f_target = sample_from_gaussian_process(
        key, batch_size=1, num_observations=n, rho=rho, sigma=sigma
    )
    return (x_target.reshape(n, 1), y_target.reshape(n, 1)), f_target.reshape(
        n, 1
    )

def _gaussian_process(**kwargs):
    kernel = ExponentiatedQuadratic()
    m = 50
    jitter = 10e-8
    gp = SparseGP(kernel, m, jitter)
    return gp(**kwargs)


def train_gp(key, x, y):
    _, init_key, train_key = random.split(key, 3)
    gaussian_process = hk.transform(_gaussian_process)
    params = gaussian_process.init(init_key, x=x, y=y)

    params, _ = train_sparse_gaussian_process(
        gaussian_process,
        params,
        train_key,
        x=x,
        y=y,
        n_iter=1000,
        stepsize=0.005
    )

    return gaussian_process, params

def plot(key, gaussian_process, params, x, y, f, train_idxs):

    x_m = params['sparse_gp']['x_m']
    y_m = jnp.zeros((x_m.shape[0],1))
    m = jnp.shape(x_m)[0]
    n = jnp.shape(train_idxs)[0]

    _, ax = plt.subplots(figsize=(15, 6))
    srt_idxs = jnp.argsort(jnp.squeeze(x))

    ax.set_title(f'Sparse GP\nTraining Points: n={n}, Inducing Points: m={m}')
    ax.plot(
        jnp.squeeze(x)[srt_idxs],
        jnp.squeeze(f)[srt_idxs],
        color="black",
        alpha=0.5,
    )
    ax.scatter(
        jnp.squeeze(x[train_idxs, :]),
        jnp.squeeze(y[train_idxs, :]),
        color="red",
        marker="+",
        alpha=0.5,
    )

    ax.scatter(
        jnp.squeeze(x_m),
        jnp.squeeze(y_m),
        color="green",
        marker="*",
        alpha=0.5,
    )

    posterior_dist = gaussian_process.apply(
        params=params,
        rng=key,
        x=x[train_idxs, :],
        y=y[train_idxs, :],
        x_star=x
    )

    y_star = posterior_dist.mean()
    ax.plot(
        jnp.squeeze(x)[srt_idxs], jnp.squeeze(y_star)[srt_idxs], color="blue"
    )

    sigma = posterior_dist.stddev()
    ucb = y_star + 1.644854 * sigma
    lcb = y_star - 1.644854 * sigma
    ax.fill_between(
        jnp.squeeze(x)[srt_idxs],
        lcb[srt_idxs], ucb[srt_idxs],
        color="grey", alpha=0.2
    )

    ax.legend(
        handles=[
            mpatches.Patch(
                color="black",
                alpha=0.5,
                label="Latent function " + r"$f \sim GP$",
            ),
            mpatches.Patch(color="red", alpha=0.45, label="Training data"),
            mpatches.Patch(color="green", alpha=0.45, label="Inducing points"),
            mpatches.Patch(color="blue", alpha=0.45, label="Posterior mean"),
            mpatches.Patch(color="grey", alpha=0.1, label=r'90% posterior interval'),
        ],
        loc="best",
        frameon=False,
    )
    ax.grid()
    ax.set_frame_on(False)
    plt.show()


def run():
    rng_seq = hk.PRNGSequence(14)
    n_train = 150

    (x, y), f = data(next(rng_seq), 0.25, 3.0)
    train_idxs = random.choice(
        next(rng_seq), jnp.arange(x.shape[0]), shape=(n_train,), replace=False
    )

    x_train, y_train = x[train_idxs, :], y[train_idxs, :]
    gaussian_process, params = train_gp(next(rng_seq), x_train, y_train)

    plot(
        next(rng_seq),
        gaussian_process,
        params,
        x,
        y,
        f,
        train_idxs
    )


if __name__ == "__main__":
    run()
