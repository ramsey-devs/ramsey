from collections import namedtuple

import chex
import haiku as hk
import jax
import numpy as np
import optax
from absl import logging
from flax.training.early_stopping import EarlyStopping
from haiku import Transformed
from jax import numpy as jnp
from jax import random
from rmsyutls import as_batch_iterator


def train_sequential_neural_process(
    fn: Transformed,  # pylint: disable=invalid-name
    params,
    seed: random.PRNGKey,
    x: np.ndarray,  # pylint: disable=invalid-name
    y: np.ndarray,  # pylint: disable=invalid-name
    n_context: int,
    n_target: int,
    optimizer=optax.adamw(1e-4),
    n_iter=20000,
    context_is_sequential=True,
    batch_size=64,
    percent_data_as_validation=0.05,
    n_early_stopping_patience=1000,
    verbose=False,
):
    data_key, seed = random.split(seed)
    train_iter = as_batch_iterator(
        rng_key=data_key,
        data=namedtuple("regression_data", "y x")(y, x),
        batch_size=batch_size,
        shuffle=False,
    )
    chex.assert_rank(y, 3)
    n_samples = y.shape[1]
    n_train = int(n_samples * (1.0 - percent_data_as_validation))

    @jax.jit
    def loss_fn(params, rng, is_train, **batch):
        _, _, lpp, kl = fn.apply(params=params, rng=rng, **batch)
        train_loss = -jnp.mean(jnp.sum(lpp[:, :n_train, :], axis=1) - kl)
        val_loss = -jnp.mean(jnp.sum(lpp[:, n_train:, :], axis=1) - kl)
        obj = jnp.where(is_train, train_loss, val_loss)
        return obj

    @jax.jit
    def step(params, state, rng, **batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, rng, True, **batch)
        updates, new_state = optimizer.update(grads, state, params)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, new_state

    rng_seq = hk.PRNGSequence(seed)
    state = optimizer.init(params)

    losses = np.zeros([n_iter, 2])
    logging.info(f"starting to train")
    early_stop = EarlyStopping(1e-5, n_early_stopping_patience)
    for i in range(n_iter):
        train_loss = 0.0
        # TODO: makes sense?
        for j in range(train_iter.num_batches):
            batch = _split_data(
                next(rng_seq),
                n_samples,
                n_context,
                n_target,
                context_is_sequential,
                **train_iter(j),
            )
            batch_loss, params, state = step(
                params, state, next(rng_seq), **batch
            )
            train_loss += batch_loss
        validation_loss = loss_fn(params, next(rng_seq), False, **batch)
        losses[i] = np.array([train_loss, validation_loss])

        if (i % 1000 == 0 or i == n_iter) - 1 and verbose:
            elbo = -float(validation_loss)
            logging.info(f"ELBO of validation set at iteration {i}: {elbo}")

        _, early_stop = early_stop.update(validation_loss)
        if early_stop.should_stop:
            logging.info(f"early stopping criterion found at itr {i}")
            break

    losses = jnp.vstack(losses)[: (i + 1), :]
    return params, losses


def _split_data(
    seed: random.PRNGKey,
    n_train: int,
    n_context: int,
    n_target: int,
    sequential_split: bool,
    x: np.ndarray,  # pylint: disable=invalid-name
    y: np.ndarray,  # pylint: disable=invalid-name
):
    assert n_train >= n_target

    if sequential_split:
        key1, key2, seed = random.split(seed, 3)
        target_idxs_start = random.choice(key1, n_train - n_target + 1)
        target_idxs = jnp.arange(
            target_idxs_start, target_idxs_start + n_target
        )
        ctx_idxs_start = random.choice(
            key2, jnp.arange(target_idxs[0], target_idxs[-1] - n_context)
        )
        ctx_idxs = jnp.arange(ctx_idxs_start, ctx_idxs_start + n_context)
    else:
        train_idxs = random.choice(
            seed, x.shape[1], shape=(n_context + n_target,), replace=False
        )
        ctx_idxs = train_idxs[:n_context]
        target_idxs = train_idxs
    x_context = x[:, ctx_idxs, :]
    y_context = y[:, ctx_idxs, :]
    x_target = x[:, target_idxs, :]
    y_target = y[:, target_idxs, :]
    return {
        "x_context": x_context,
        "y_context": y_context,
        "x_target": x_target,
        "y_target": y_target,
    }
