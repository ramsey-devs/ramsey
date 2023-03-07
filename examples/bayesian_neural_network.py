"""
Bayesian Neural Network
=======================

This example implements the training and prediction of a
Bayesian neural network. Predictions from a Haiku MLP from
the same data are shown as a reference.

References
----------

[1] Blundell C., Cornebise J., Kavukcuoglu K., Wierstra D.
    "Weight Uncertainty in Neural Networks".
    ICML, 2015.
"""

import haiku as hk
import jax
import matplotlib.pyplot as plt
import optax
from jax import numpy as jnp
from jax import random

from ramsey.contrib import BNN, BayesianLinear
from ramsey.data import sample_from_linear_model


def data(key):
    (x_target, y_target), f_target = sample_from_linear_model(
        key, batch_size=1, num_observations=50, noise_scale=1.0
    )

    return (x_target.reshape(-1, 1), y_target.reshape(-1, 1)), f_target.reshape(
        -1, 1
    )


def _bayesian_nn(**kwargs):
    layers = [
        BayesianLinear(8, with_bias=True),
        hk.Linear(2, with_bias=False),
    ]
    bnn = BNN(layers)
    return bnn(**kwargs)


def train_bnn(
    rng_seq,
    x,  # pylint: disable=invalid-name
    y,  # pylint: disable=invalid-name
    n_iter=20000,
    stepsize=0.001,
):
    bnn = hk.transform(_bayesian_nn)
    params = bnn.init(next(rng_seq), x=x, y=y)

    @jax.jit
    def step(params, opt_state, rng, x, y):
        def _loss(params):
            _, loss = bnn.apply(params, rng, x=x, y=y)
            return loss

        obj, grads = jax.value_and_grad(_loss)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, obj

    optimizer = optax.adam(stepsize)
    opt_state = optimizer.init(params)

    losses = [0] * n_iter
    for i in range(n_iter):
        params, opt_state, loss = step(params, opt_state, next(rng_seq), x, y)
        losses[i] = float(loss)
        if i % 200 == 0 or i == n_iter - 1:
            elbo = -float(loss)
            print(f"ELBO at {i}: {elbo}")
    return bnn, params


def plot(rng_seq, bnn, params, x, f, x_train, y_train):
    _, ax = plt.subplots(figsize=(8, 3))
    srt_idxs = jnp.argsort(jnp.squeeze(x))
    for i in range(20):
        posterior = bnn.apply(params=params, rng=next(rng_seq), x=x)
        y = posterior.sample(next(rng_seq))
        ax.plot(
            jnp.squeeze(x)[srt_idxs],
            jnp.squeeze(y)[srt_idxs],
            color="grey",
            alpha=0.1,
        )
    ax.scatter(
        jnp.squeeze(x_train),
        jnp.squeeze(y_train),
        color="blue",
        marker=".",
        alpha=0.75,
    )
    ax.plot(jnp.squeeze(x), jnp.squeeze(f), color="blue", alpha=0.75)
    ax.grid()
    ax.set_frame_on(False)
    plt.show()


def choose_training_samples(key, x, y, n_train):
    train_idxs = random.choice(
        key, jnp.arange(x.shape[0]), shape=(n_train, 1), replace=False
    )
    x_train, y_train = jnp.take(x, train_idxs), jnp.take(y, train_idxs)
    return x_train, y_train


def run():
    rng_seq = hk.PRNGSequence(23)
    n_train = 40

    (x, y), f = data(next(rng_seq))
    x_train, y_train = choose_training_samples(next(rng_seq), x, y, n_train)
    bnn, params = train_bnn(rng_seq, x=x_train, y=y_train)
    plot(rng_seq, bnn, params, x, f, x_train, y_train)


if __name__ == "__main__":
    run()
