from collections import namedtuple

import jax
import numpy as np
import optax
from flax.training.train_state import TrainState
from jax import random as jr, Array
from rmsyutls import as_batch_iterator
from tqdm import tqdm
from ramsey._src.experimental.bayesian_neural_network.bayesian_neural_network import \
    BNN


__all__ = ["train_bnn"]


def _create_train_state(rng, model, optimizer, **init_data):
    init_key, sample_key = jr.split(rng)
    params = model.init({"sample": sample_key, "params": init_key}, **init_data)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    return state


def train_bnn(
        rng_key,
        bnn: BNN,
        x: Array,
        y: Array,
        optimizer=optax.adam(3e-4),
        n_iter=10000,
        batch_size=128,
        verbose=False
):
    itr_key, seed = jr.split(rng_key)
    train_itr = as_batch_iterator(
        itr_key, namedtuple("data", "y x")(y, x), batch_size, True
    )

    init_key, seed = jr.split(seed)
    state = _create_train_state(init_key, bnn, optimizer, **train_itr(0))

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
        if (i % 1000 == 0 or i == n_iter - 1) and verbose:
            elbo = -float(objective)
            print(f"ELBO at {i}: {elbo}")

    return state.params, objectives
