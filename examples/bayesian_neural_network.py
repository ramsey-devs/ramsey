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
import warnings
from collections import namedtuple

import numpy as np
from flax import linen as nn
import jax
import matplotlib.pyplot as plt
import optax
from flax.training.train_state import TrainState
from jax import numpy as jnp, random as jr
from rmsyutls import as_batch_iterator
from tqdm import tqdm

from ramsey.contrib import BNN, BayesianLinear
from ramsey.data import sample_from_gaussian_process


def data(key):
    data = sample_from_gaussian_process(
        key, batch_size=1, num_observations=1000
    )
    return (
        (data.x.reshape(-1, 1), data.y.reshape(-1, 1)),
        data.f.reshape(-1, 1)
    )


def get_bayesian_nn():
    layers = [
        BayesianLinear(128, with_bias=True),
        BayesianLinear(128, with_bias=True),
        nn.Dense(2, with_bias=False),
    ]
    bnn = BNN(layers)
    return bnn


def create_train_state(rng, model, **init_data):
    init_key, sample_key = jr.split(rng)
    optimizer = optax.adam(3e-4)
    params = model.init({'sample': sample_key, 'params': init_key}, **init_data)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    return state


def train(seed, bnn, x, y, n_iter=1000):
    itr_key, seed = jr.split(seed)
    train_itr = as_batch_iterator(
       itr_key, namedtuple("data", "y x")(y, x), 64, True
    )

    init_key, seed = jr.split(seed)
    state = create_train_state(init_key, bnn, **train_itr(0))

    @jax.jit
    def step(rngs, state, **batch):
        step = state.step
        rngs = {name: jr.fold_in(rng, step) for name, rng in rngs.items()}

        def obj_fn(params):
            _, loss = bnn.apply(variables=params, rngs=rngs, **batch)
            return loss

        obj, grads = jax.value_and_grad(obj_fn)(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, obj

    objectives = np.zeros(n_iter)
    for i in tqdm(range(n_iter)):
        objective = 0
        for j in range(train_itr.num_batches):
            batch = train_itr(j)
            sample_rng_key, seed = jr.split(seed)
            state, train_loss = step({"sample": sample_rng_key}, state, **batch)
            objective += train_loss
        objectives[i] = objective
        if i % 200 == 0 or i == n_iter - 1:
            elbo = -float(objective)
            print(f"ELBO at {i}: {elbo}")

    return state.params, objectives


def plot(seed, bnn, params, x, f, x_train, y_train):
    _, ax = plt.subplots(figsize=(8, 3))
    srt_idxs = jnp.argsort(jnp.squeeze(x))
    for i in range(20):
        rng_key, sample_key, seed = jr.split(seed, 3)
        posterior = bnn.apply(variables=params, rngs={'sample': rng_key}, x=x)
        y = posterior.sample(sample_key)
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


def sample_training_points(key, x, y, n_train):
    train_idxs = jr.choice(
        key, jnp.arange(x.shape[0]), shape=(n_train,), replace=False
    )
    return x[train_idxs], y[train_idxs]


def run():
    warnings.warn(
        "the BNN is labelled 'experimental'. "
        "experimental code is hardly tested or debugged"
    )
    data_rng_key, sample_rng_key, seed = jr.split(jr.PRNGKey(0), 3)
    (x, y), f = data(data_rng_key)
    x_train, y_train = sample_training_points(sample_rng_key, x, y, 900)

    train_rng_key, seed = jr.split(seed)
    bnn = get_bayesian_nn()
    bnn, params = train(train_rng_key, bnn, x=x_train, y=y_train)

    plot_rng_key, seed = jr.split(seed)
    plot(plot_rng_key, bnn, params, x, f, x_train, y_train)


if __name__ == "__main__":
    run()
