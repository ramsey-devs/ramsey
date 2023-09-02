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
import matplotlib.patches as mpatches
from flax import linen as nn
import jax
import matplotlib.pyplot as plt
import optax
from flax.training.train_state import TrainState
from jax import numpy as jnp, random as jr
from rmsyutls import as_batch_iterator
from tqdm import tqdm

from ramsey.experimental import BNN, BayesianLinear
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


def create_train_state(rng, model, **init_data):
    init_key, sample_key = jr.split(rng)

    boundary = 1000
    warmup_schedule = optax.linear_schedule(
        init_value=0.0001, end_value=0.001, transition_steps=boundary
    )
    decay_schedule = optax.exponential_decay(
        decay_rate=0.9,
        init_value=0.001,
        end_value=0.0001,
        transition_steps=10000,
    )
    schedule = optax.join_schedules(
        [warmup_schedule, decay_schedule],
        boundaries=[boundary]
    )
    optimizer = optax.adam(schedule)

    params = model.init({'sample': sample_key, 'params': init_key}, **init_data)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    return state


def train(seed, bnn, x, y, n_iter=10000):
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
        if i % 1000 == 0 or i == n_iter - 1:
            elbo = -float(objective)
            print(f"ELBO at {i}: {elbo}")

    return state.params, objectives


def plot(seed, bnn, params, x, f, x_train, y_train):
    _, ax = plt.subplots(figsize=(10, 4))
    srt_idxs = jnp.argsort(jnp.squeeze(x))
    ys = []
    for i in range(100):
        rng_key, sample_key, seed = jr.split(seed, 3)
        posterior = bnn.apply(variables=params, rngs={'sample': rng_key}, x=x)
        y = posterior.sample(sample_key)
        ys.append(y)
    yhat = jnp.hstack(ys).T
    yhat_mean = jnp.mean(yhat, axis=0)
    y_hat_cis = jnp.quantile(yhat, q=jnp.array([0.05, 0.95]), axis=0)
    ax.plot(
        jnp.squeeze(x)[srt_idxs],
        jnp.squeeze(yhat_mean)[srt_idxs],
        color="#011482",
        alpha=0.9,
    )
    ax.fill_between(
        np.squeeze(x),
        y_hat_cis[0],
        y_hat_cis[1],
        color="#011482",
        alpha=0.2
    )
    ax.scatter(
        jnp.squeeze(x_train),
        jnp.squeeze(y_train),
        color="black",
        marker=".",
        s=1
    )
    ax.plot(jnp.squeeze(x), jnp.squeeze(f), color="black", alpha=0.5)
    ax.legend(
        handles=[
            mpatches.Patch(color="black", label="Training data"),
            mpatches.Patch(color="#011482", label="Posterior mean", alpha=0.9),
            mpatches.Patch(color="#011482", label="90% posterior intervals", alpha=0.2),
        ],
        bbox_to_anchor=(1.025, 0.6), frameon=False,
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
    warnings.warn(
        "The BNN is labelled as 'experimental'. "
        "Experimental code is hardly tested or debugged."
    )
    data_rng_key, sample_rng_key, seed = jr.split(jr.PRNGKey(0), 3)
    (x, y), f = data(data_rng_key)
    x_train, y_train = sample_training_points(sample_rng_key, x, y, 900)

    train_rng_key, seed = jr.split(seed)
    bnn = get_bayesian_nn()
    params, objectives = train(train_rng_key, bnn, x=x_train, y=y_train)

    plot_rng_key, seed = jr.split(seed)
    plot(plot_rng_key, bnn, params, x, f, x_train, y_train)


if __name__ == "__main__":
    run()
