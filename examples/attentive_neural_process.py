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

import haiku as hk
from jax import numpy as jnp, random
import matplotlib.pyplot as plt

from ramsey.train import train_neural_process
from ramsey.attention import MultiHeadAttention
from ramsey.data import sample_from_gaussian_process
from ramsey.models import ANP


def data(key):
    (x_target, y_target), f_target = sample_from_gaussian_process(
        key, batch_size=10, num_observations=200
    )
    return (x_target, y_target), f_target


def _neural_process(**kwargs):
    dim = 3
    np = ANP(
        decoder=hk.nets.MLP([dim] * 3 + [2]),
        latent_encoder=(hk.nets.MLP([dim] * 3), hk.nets.MLP([dim, dim * 2])),
        deterministic_encoder=(
            hk.nets.MLP([dim] * 3),
            MultiHeadAttention(
                num_heads=8, head_size=16, embedding=hk.nets.MLP([dim] * 2)
            ),
        ),
    )
    return np(**kwargs)


def train_np(key, n_context, n_target, x_target, y_target):
    _, init_key, train_key = random.split(key, 3)
    neural_process = hk.transform(_neural_process)
    params = neural_process.init(
        init_key, x_context=x_target, y_context=y_target, x_target=x_target
    )

    params, _ = train_neural_process(
        neural_process,
        params,
        train_key,
        x=x_target,
        y=y_target,
        n_context=n_context,
        n_target=n_target,
        n_iter=10000,
    )

    return neural_process, params


def plot(
    key,
    neural_process,
    params,
    x_target,
    y_target,
    f_target,
    n_context,
    n_target,
):
    key, sample_key = random.split(key, 2)
    sample_idxs = random.choice(
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

        for i in range(20):
            key, apply_key = random.split(key, 2)
            y_star = neural_process.apply(
                params=params,
                rng=apply_key,
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


def run():
    rng_seq = hk.PRNGSequence(12)
    n_context, n_target = 10, 20

    (x_target, y_target), f_target = data(next(rng_seq))
    neural_process, params = train_np(
        next(rng_seq), n_context, n_target, x_target, y_target
    )
    plot(
        next(rng_seq),
        neural_process,
        params,
        x_target,
        y_target,
        f_target,
        n_context,
        n_target,
    )


if __name__ == "__main__":
    run()
