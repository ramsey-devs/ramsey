"""
Bayesian neural network
=======================

This example implements the training and prediction of a
Bayesian neural network. Predictions from a Haiku MLP from
the same data are shown as a reference.

References
----------
[1] Blundell C., Cornebise J., Kavukcuoglu K., Wierstra D.
    "Weight Uncertainty in Neural Networks". ICML, 2015.
"""

import argparse
import warnings

import jax
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from flax import linen as nn
from jax import numpy as jnp
from jax import random as jr

from ramsey.data import sample_from_gaussian_process
from ramsey.experimental import BNN, BayesianLinear, train_bnn


def data(key, n_samples):
    data = sample_from_gaussian_process(
        key, batch_size=1, num_observations=n_samples
    )
    return (
        (data.x.reshape(-1, 1), data.y.reshape(-1, 1)),
        data.f.reshape(-1, 1),
    )


def get_bayesian_nn():
    layers = [
        BayesianLinear(16, use_bias=False, mc_sample_size=10),
        jax.nn.relu,
        nn.Dense(128, use_bias=True),
        jax.nn.relu,
        BayesianLinear(16, use_bias=False, mc_sample_size=10),
        jax.nn.relu,
        nn.Dense(2, use_bias=True),
    ]
    bnn = BNN(layers)
    return bnn


def plot(seed, bnn, params, x, f, x_train, y_train):
    srt_idxs = jnp.argsort(jnp.squeeze(x))
    ys = []
    for i in range(100):
        rng_key, sample_key, seed = jr.split(seed, 3)
        posterior = bnn.apply(variables=params, rngs={"sample": rng_key}, x=x)
        y = posterior.sample(sample_key)
        ys.append(y)
    yhat = jnp.hstack(ys).T
    yhat_mean = jnp.mean(yhat, axis=0)
    y_hat_cis = jnp.quantile(yhat, q=jnp.array([0.05, 0.95]), axis=0)

    _, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        jnp.squeeze(x)[srt_idxs],
        jnp.squeeze(yhat_mean)[srt_idxs],
        color="#011482",
        alpha=0.9,
    )
    ax.fill_between(
        np.squeeze(x), y_hat_cis[0], y_hat_cis[1], color="#011482", alpha=0.2
    )
    ax.scatter(
        jnp.squeeze(x_train),
        jnp.squeeze(y_train),
        color="black",
        marker=".",
        s=1,
    )
    ax.plot(jnp.squeeze(x), jnp.squeeze(f), color="black", alpha=0.5)
    ax.legend(
        handles=[
            mpatches.Patch(color="black", label="Training data"),
            mpatches.Patch(color="#011482", label="Posterior mean", alpha=0.9),
            mpatches.Patch(
                color="#011482", label="90% posterior intervals", alpha=0.2
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


def run(args):
    warnings.warn(
        "The BNN is labelled as 'experimental'. "
        "Experimental code is hardly tested or debugged."
    )
    data_rng_key, sample_rng_key, seed = jr.split(jr.PRNGKey(0), 3)
    n_samples = 200
    (x, y), f = data(data_rng_key, n_samples)
    x_train, y_train = sample_training_points(
        sample_rng_key, x, y, int(n_samples * 0.9)
    )

    train_rng_key, seed = jr.split(seed)
    bnn = get_bayesian_nn()
    params, objectives = train_bnn(
        train_rng_key,
        bnn,
        x_train,
        y_train,
        n_iter=args.num_iter,
        batch_size=64,
    )

    plot_rng_key, seed = jr.split(seed)
    plot(plot_rng_key, bnn, params, x, f, x_train, y_train)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "---num_iter", type=int, default=50000)
    run(parser.parse_args())
