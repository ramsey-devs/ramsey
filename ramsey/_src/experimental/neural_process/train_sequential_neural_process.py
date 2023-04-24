import logging
from collections import namedtuple

import haiku as hk
import jax
import optax
import rmsyutls
from flax.training.early_stopping import EarlyStopping
from haiku import Transformed
from haiku._src.data_structures import FlatMapping
from jax import numpy as jnp
from jax import numpy as np
from jax import random

# pylint: disable=too-many-locals
from rmsyutls import as_batch_iterator, as_batch_iterators


def train_sequential_neural_process(
    fn: Transformed,  # pylint: disable=invalid-name
    params: FlatMapping,
    seed: random.PRNGKey,
    x: np.ndarray,  # pylint: disable=invalid-name
    y: np.ndarray,  # pylint: disable=invalid-name
    n_context: int,
    optimizer=optax.adam(3e-4),
    n_iter=20000,
    context_is_sequential=True,
    batch_size=64,
    percent_data_as_validation=0.05,
    n_early_stopping_patience=20,
    verbose=False,
):
    data_key, seed = random.split(seed)
    train_iter = as_batch_iterator(
        rng_key=data_key,
        data=namedtuple("regression data", "y x")(y, x),
        batch_size=batch_size,
        shuffle=False,
    )
    n_samples = y.shape[0]
    n_train = int(n_samples) * (1.0 - percent_data_as_validation)

    @jax.jit
    def loss(params, rng, is_train, **batch):
        _, obj = fn.apply(params=params, rng=rng, **batch)
        if is_train:
            return jnp.mean(obj[:n_train])
        else:
            return jnp.mean(obj[n_train:])

    @jax.jit
    def step(params, state, rng, **batch):
        loss, grads = jax.value_and_grad(loss)(params, rng, True, **batch)
        updates, new_state = optimizer.update(grads, state)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, new_state

    rng_seq = hk.PRNGSequence(seed)
    state = optimizer.init(params)

    losses = np.zeros([n_iter, 2])
    early_stop = EarlyStopping(1e-3, n_early_stopping_patience)
    for i in range(n_iter):
        train_loss = 0.0
        for j in range(train_iter.num_batches):
            batch = _split_data(
                next(rng_seq), n_context, context_is_sequential, **train_iter(j)
            )
            batch_loss, params, state = step(params, state, **batch)
            train_loss += batch_loss
        validation_loss = loss(params, next(rng_seq), False, **batch)
        losses[i] = jnp.array([train_loss, validation_loss])

        if (i % 1000 == 0 or i == n_iter) - 1 and verbose:
            elbo = -float(validation_loss)
            print(f"ELBO of validation set at iteration {i}: {elbo}")

        _, early_stop = early_stop.update(validation_loss)
        if early_stop.should_stop:
            logging.info("early stopping criterion found")
            break

    losses = jnp.vstack(losses)[: (i + 1), :]
    return params, losses


def _split_data(
    key: random.PRNGKey,
    n_context: int,
    sequential_split: bool,
    x: np.ndarray,  # pylint: disable=invalid-name
    y: np.ndarray,  # pylint: disable=invalid-name
):
    if sequential_split:
        context_idx_start = random.choice(key, x.shape[1] - n_context)
        context_idxs = jnp.arange(
            context_idx_start, context_idx_start + n_context
        )
    else:
        context_idxs = random.choice(
            key, x.shape[1], shape=(n_context,), replace=False
        )
    x_context = x[:, context_idxs:]
    y_context = y[:, context_idxs, :]
    return {
        "x_context": x_context,
        "y_context": y_context,
        "x_target": x,
        "y_target": y,
    }
