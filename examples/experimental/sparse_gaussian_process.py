"""
Sparse Gaussian process regression
==================================

This example implements the training and prediction of a sparse Gaussian process
regression model.

References
----------

[1] Titsias, Michalis K.
    "Variational Learning of Inducing Variables in Sparse Gaussian Processes".
    AISTATS, 2009.
"""


import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from jax import numpy as jnp
from jax import random as jr
from jax.config import config

from ramsey.data import sample_from_gaussian_process
from ramsey.experimental import (
    ExponentiatedQuadratic,
    SparseGP,
    train_sparse_gaussian_process,
)

config.update("jax_enable_x64", True)


def data(key, rho, sigma, n=1000):
    data = sample_from_gaussian_process(
        key, batch_size=1, num_observations=n, rho=rho, sigma=sigma
    )
    return (
        (data.x.reshape(-1, 1), data.y.reshape(-1, 1)),
        data.f.reshape(-1, 1),
    )


def get_gaussian_process():
    gp = SparseGP(ExponentiatedQuadratic(), n_inducing=50)
    return gp


def train(rng_key, x, y):
    gaussian_process = get_gaussian_process()
    params, _ = train_sparse_gaussian_process(
        rng_key,
        gaussian_process,
        x=x,
        y=y,
    )

    return gaussian_process, params


def plot(seed, gaussian_process, params, x, y, f, x_train, y_train):
    srt_idxs = jnp.argsort(jnp.squeeze(x))
    x_m = params["params"]["x_inducing"]
    y_m = jnp.zeros((x_m.shape[0], 1))

    apply_key, seed = jr.split(seed, 2)
    posterior_dist = gaussian_process.apply(
        variables=params,
        rngs={"sample": apply_key},
        x=x_train,
        y=y_train,
        x_star=x,
    )

    sample_key, seed = jr.split(seed, 2)
    yhat = posterior_dist.sample(sample_key, (100,))
    yhat_mean = jnp.mean(yhat, axis=0)
    y_hat_cis = jnp.quantile(yhat, q=jnp.array([0.05, 0.95]), axis=0)

    _, ax = plt.subplots(figsize=(15, 6))
    ax.scatter(
        jnp.squeeze(x_train),
        jnp.squeeze(y_train),
        color="black",
        marker="+",
    )
    ax.scatter(
        jnp.squeeze(x_m),
        jnp.squeeze(y_m),
        color="red",
        marker="+",
    )
    ax.plot(
        jnp.squeeze(x)[srt_idxs],
        jnp.squeeze(f)[srt_idxs],
        color="grey",
    )
    ax.plot(
        jnp.squeeze(x)[srt_idxs],
        jnp.squeeze(yhat_mean)[srt_idxs],
        color="#011482",
        alpha=0.9,
    )
    ax.fill_between(
        jnp.squeeze(x),
        y_hat_cis[0],
        y_hat_cis[1],
        color="#011482",
        alpha=0.2,
    )
    ax.legend(
        handles=[
            mpatches.Patch(color="black", label="Training data"),
            mpatches.Patch(color="red", label="Inducing points"),
            mpatches.Patch(color="grey", label="Latent GP"),
            mpatches.Patch(color="#204a87", label="Posterior mean", alpha=0.9),
            mpatches.Patch(
                color="#204a87", label="90% posterior intervals", alpha=0.2
            ),
        ],
        bbox_to_anchor=(1.025, 0.6),
        frameon=False,
    )
    ax.grid()
    ax.set_frame_on(False)
    plt.tight_layout()
    plt.show()


def sample_training_points(key, x, y, n_train):
    train_idxs = jr.choice(
        key, jnp.arange(x.shape[0]), shape=(n_train,), replace=False
    )
    return x[train_idxs], y[train_idxs]


def run():
    data_rng_key, sample_rng_key, seed = jr.split(jr.PRNGKey(0), 3)
    (x, y), f = data(data_rng_key, 0.25, 3.0)
    x_train, y_train = sample_training_points(sample_rng_key, x, y, 100)

    train_rng_key, seed = jr.split(seed)
    gaussian_process, params = train(train_rng_key, x=x_train, y=y_train)

    plot_rng_key, seed = jr.split(seed)
    plot(plot_rng_key, gaussian_process, params, x, y, f, x_train, y_train)


if __name__ == "__main__":
    run()
