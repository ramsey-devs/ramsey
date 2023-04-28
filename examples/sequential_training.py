"""
Training for sequential neural processes
=======================================

Here, we show how neural processes that have sequential/recurrent decoders
can be trained alternatively.

"""

import argparse

import haiku as hk
import jax
import matplotlib.patches as mpatches
import numpy as np
import optax
from absl import logging
from jax import numpy as jnp
from jax import random
from matplotlib import pyplot as plt

from ramsey.attention import MultiHeadAttention
from ramsey.contrib import RANP
from ramsey.data import sample_from_gaussian_process
from ramsey.experimental import (
    RDANP,
    Recurrent,
    RecurrentEncodingRANP,
    RecurrentEncodingRNP,
)
from ramsey.experimental.train import train_sequential_neural_process

logging.set_verbosity(logging.INFO)


def build_recurrent():
    def f(**kwargs):
        np = Recurrent(
            decoder=hk.DeepRNN(
                [
                    hk.LSTM(hidden_size=20),
                    jax.nn.tanh,
                    hk.LSTM(hidden_size=20),
                    jax.nn.tanh,
                    hk.Linear(2),
                ]
            )
        )
        return np(**kwargs)

    return f


def build_ranp():
    dim = 32

    def f(**kwargs):
        np = RANP(
            decoder=hk.DeepRNN(
                [
                    hk.LSTM(hidden_size=20),
                    jax.nn.tanh,
                    hk.LSTM(hidden_size=20),
                    jax.nn.tanh,
                    hk.Linear(2),
                ]
            ),
            latent_encoder=(
                hk.nets.MLP([dim] * 2),
                hk.nets.MLP([dim, dim * 2]),
            ),
            deterministic_encoder=(
                hk.nets.MLP([dim] * 2),
                MultiHeadAttention(
                    num_heads=4, head_size=32, embedding=hk.nets.MLP([dim] * 2)
                ),
            ),
        )
        return np(**kwargs)

    return f


def build_rdanp():
    dim = 32

    def f(**kwargs):
        np = RDANP(
            decoder=hk.DeepRNN(
                [
                    hk.LSTM(hidden_size=20),
                    jax.nn.tanh,
                    hk.LSTM(hidden_size=20),
                    jax.nn.tanh,
                    hk.Linear(2),
                ]
            ),
            latent_encoder=(
                hk.nets.MLP([dim] * 2),
                MultiHeadAttention(num_heads=2, head_size=32),
                hk.nets.MLP([dim, dim * 2]),
            ),
            deterministic_encoder=(
                hk.nets.MLP([dim] * 2),
                MultiHeadAttention(num_heads=2, head_size=32),
                MultiHeadAttention(
                    num_heads=4, head_size=32, embedding=hk.nets.MLP([dim] * 2)
                ),
            ),
        )
        return np(**kwargs)

    return f


def build_re_rnp():
    dim = 32

    def f(**kwargs):
        np = RecurrentEncodingRNP(
            decoder=hk.DeepRNN(
                [
                    hk.LSTM(hidden_size=20),
                    jax.nn.tanh,
                    hk.LSTM(hidden_size=20),
                    jax.nn.tanh,
                    hk.Linear(2),
                ]
            ),
            latent_encoder=(
                hk.DeepRNN(
                    [
                        hk.LSTM(hidden_size=10),
                        jax.nn.tanh,
                        hk.LSTM(hidden_size=10),
                    ]
                ),
                hk.nets.MLP([dim, dim * 2]),
            ),
            deterministic_encoder=hk.DeepRNN(
                [
                    hk.LSTM(hidden_size=10),
                    jax.nn.tanh,
                    hk.LSTM(hidden_size=10),
                ]
            ),
        )
        return np(**kwargs)

    return f


def build_re_ranp():
    dim = 32

    def f(**kwargs):
        np = RecurrentEncodingRANP(
            decoder=hk.DeepRNN(
                [
                    hk.LSTM(hidden_size=20),
                    jax.nn.tanh,
                    hk.LSTM(hidden_size=20),
                    jax.nn.tanh,
                    hk.Linear(2),
                ]
            ),
            latent_encoder=(
                hk.DeepRNN(
                    [
                        hk.LSTM(hidden_size=10),
                        jax.nn.tanh,
                        hk.LSTM(hidden_size=10),
                    ]
                ),
                hk.nets.MLP([dim, dim * 2]),
            ),
            deterministic_encoder=(
                hk.DeepRNN(
                    [
                        hk.LSTM(hidden_size=10),
                        jax.nn.tanh,
                        hk.LSTM(hidden_size=10),
                        jax.nn.tanh,
                        hk.Linear(2),
                    ]
                ),
                MultiHeadAttention(
                    num_heads=4, head_size=32, embedding=hk.nets.MLP([dim] * 2)
                ),
            ),
        )
        return np(**kwargs)

    return f


def get_model(model):
    return {
        "Recurrent": build_recurrent(),
        "RANP": build_ranp(),
        "RDANP": build_rdanp(),
        "RecurrentEncodingRNP": build_re_rnp(),
        "RecurrentEncodingRANP": build_re_ranp(),
    }[model]


def data(key, use_discrete_data, len_timeseries):
    (x_target, y_target), f_target = sample_from_gaussian_process(
        key, batch_size=4, num_observations=len_timeseries
    )
    return (x_target, y_target), f_target


def _train_np_with_sequential_structure(
    rng_seq, model, n_context, n_target, x_target, y_target
):
    neural_process = hk.transform(model)
    params = neural_process.init(
        next(rng_seq),
        x_context=x_target,
        y_context=y_target,
        x_target=x_target,
        y_target=y_target,
    )

    optimizer = optax.adamw(
        optax.linear_schedule(
            init_value=1e-03, end_value=3e-05, transition_steps=20000
        )
    )
    params, losses = train_sequential_neural_process(
        neural_process,
        params,
        next(rng_seq),
        x=x_target,
        y=y_target,
        n_context=n_context,
        n_target=n_target,
        n_iter=10000,
        batch_size=32,
        n_early_stopping_patience=1000,
        optimizer=optimizer,
    )

    plt.plot(losses)
    plt.show()

    return neural_process, params


def _plot(
    rng_seq,
    model_str,
    neural_process,
    params,
    x_target,
    y_target,
    f_target,
    n_context,
    n_target,
    n_train,
):
    context_idx_start = random.choice(next(rng_seq), n_target - n_context)
    context_idxs = jnp.arange(context_idx_start, context_idx_start + n_context)

    idxs = [0, 1, 2, 3]
    _, axes = plt.subplots(figsize=(10, 6), nrows=2, ncols=2)
    for _, (idx, ax) in enumerate(zip(idxs, axes.flatten())):
        time = np.squeeze(x_target[idx, :, 0])
        f = np.squeeze(f_target[idx, :, :])
        y = np.squeeze(y_target[idx, :, :])
        srt_idxs = np.argsort(time)
        ax.scatter(
            time[:n_train],
            y[:n_train],
            color="gray",
            marker=".",
            alpha=0.75,
        )
        ax.scatter(
            time[n_train:],
            y[n_train:],
            color="red",
            marker=".",
            alpha=0.75,
        )
        ax.plot(
            time[srt_idxs],
            f[srt_idxs],
            color="black",
            alpha=0.75,
            linestyle="--",
        )
        ax.scatter(
            time[context_idxs],
            y[context_idxs],
            color="black",
            marker="+",
            alpha=0.75,
        )
        for _ in range(10):
            if neural_process is None:
                continue
            y_star = neural_process.apply(
                params=params,
                rng=next(rng_seq),
                x_context=x_target[
                    [idx],
                    [context_idxs],
                    :,
                ],
                y_context=y_target[[idx], [context_idxs], :],
                x_target=x_target[[idx], :, :],
            ).mean
            time_star = np.squeeze(x_target[[idx], :, 0])
            y_star = np.squeeze(y_star)
            ax.plot(
                time_star[srt_idxs], y_star[srt_idxs], color="blue", alpha=0.1
            )
    ax.legend(
        handles=[
            mpatches.Patch(
                color="black",
                alpha=0.5,
                label="Context points",
            ),
            mpatches.Patch(color="blue", alpha=0.45, label="Prediction"),
            mpatches.Patch(
                color="grey", alpha=0.1, label=r"Training/validation data"
            ),
            mpatches.Patch(color="red", alpha=0.1, label=r"Test data"),
        ],
        bbox_to_anchor=(1.55, 1.15),
        frameon=False,
    )
    plt.suptitle(f"Model: {model_str}")
    plt.show()


def run(model_str):
    model = get_model(model_str)
    rng_seq = hk.PRNGSequence(1234)

    len_timeseries = 200
    (x_target, y_target), f_target = data(rng_seq, False, len_timeseries)
    n_context, n_target, n_train = 30, 195, 195

    neural_process, params = _train_np_with_sequential_structure(
        rng_seq,
        model,
        n_context,
        n_target,
        x_target[:, :n_train, :],
        y_target[:, :n_train, :],
    )

    _plot(
        rng_seq,
        model_str,
        neural_process,
        params,
        x_target,
        y_target,
        f_target,
        n_context,
        len_timeseries,
        n_train,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=[
            "RANP",
            "RDANP",
            "RecurrentEncodingRNP",
            "RecurrentEncodingRANP",
            "Recurrent",
        ],
        default="RecurrentEncodingRANP",
    )
    args = parser.parse_args()
    run(args.model)
