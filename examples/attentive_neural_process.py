"""
Attentive neural process
========================

Here, we implement and train an attentive neural process
and visualize predictions thereof.

References
----------

[1] Kim, Hyunjik, et al. "Attentive Neural Processes."
    International Conference on Learning Representations. 2019.
"""
import argparse

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from jax import numpy as jnp
from jax import random as jr

from ramsey import ANP, train_neural_process
from ramsey.data import sample_from_gaussian_process
from ramsey.nn import MLP, MultiHeadAttention


def data(key):
    data = sample_from_gaussian_process(
        key, batch_size=10, num_observations=200
    )
    return (data.x, data.y), data.f


def get_neural_process():
    dim = 128
    np = ANP(
        decoder=MLP([dim] * 3 + [2]),
        latent_encoder=(MLP([dim] * 3), MLP([dim, dim * 2])),
        deterministic_encoder=(
            MLP([dim] * 3),
            MultiHeadAttention(
                num_heads=8, head_size=16, embedding=MLP([dim] * 2)
            ),
        ),
    )
    return np


def train_np(key, n_context, n_target, x_target, y_target, num_iter):
    neural_process = get_neural_process()
    params, _ = train_neural_process(
        key,
        neural_process,
        x=x_target,
        y=y_target,
        n_context=n_context,
        n_target=n_target,
        n_iter=num_iter,
        batch_size=2,
    )
    return neural_process, params


def plot(
    seed,
    neural_process,
    params,
    x_target,
    y_target,
    f_target,
    n_context,
    n_target,
):
    sample_key, seed = jr.split(seed)
    sample_idxs = jr.choice(
        sample_key,
        x_target.shape[1],
        shape=(n_context + n_target,),
        replace=False,
    )

    idxs = [0, 2, 5, 7]
    _, axes = plt.subplots(figsize=(10, 6), nrows=2, ncols=2)
    for _, (idx, ax) in enumerate(zip(idxs, axes.flatten())):
        x = jnp.squeeze(x_target[idx, :, :])
        f = jnp.squeeze(f_target[idx, :, :])
        y = jnp.squeeze(y_target[idx, :, :])

        srt_idxs = jnp.argsort(x)
        ax.plot(x[srt_idxs], f[srt_idxs], color="blue", alpha=0.75)
        ax.scatter(
            x[sample_idxs[:n_context]],
            y[sample_idxs[:n_context]],
            color="blue",
            marker="+",
            alpha=0.75,
        )

        for _ in range(20):
            sample_rng_key, seed = jr.split(seed, 2)
            y_star = neural_process.apply(
                variables=params,
                rngs={"sample": sample_rng_key},
                x_context=x[jnp.newaxis, sample_idxs, jnp.newaxis],
                y_context=y[jnp.newaxis, sample_idxs, jnp.newaxis],
                x_target=x_target[[idx], :, :],
            ).mean
            x_star = jnp.squeeze(x_target[[idx], :, :])
            y_star = jnp.squeeze(y_star)
            ax.plot(
                x_star[srt_idxs], y_star[srt_idxs], color="black", alpha=0.1
            )
        ax.grid()
        ax.set_frame_on(False)
    plt.show()


def run(args):
    n_context, n_target = (5, 10), (20, 30)
    data_rng_key, train_rng_key, plot_rng_key = jr.split(jr.PRNGKey(0), 3)
    (x_target, y_target), f_target = data(data_rng_key)

    neural_process, params = train_np(
        train_rng_key, n_context, n_target, x_target, y_target, args.num_iter
    )

    plot(
        plot_rng_key,
        neural_process,
        params,
        x_target,
        y_target,
        f_target,
        n_context=10,
        n_target=20,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "---num_iter", type=int, default=10000)
    run(parser.parse_args())
