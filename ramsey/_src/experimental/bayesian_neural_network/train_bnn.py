from collections import namedtuple

import jax
import numpy as np
import optax
from flax.training.train_state import TrainState
from jax import Array
from jax import random as jr
from rmsyutls import as_batch_iterator
from tqdm import tqdm

# pylint: disable=line-too-long
from ramsey._src.experimental.bayesian_neural_network.bayesian_neural_network import (
    BNN,
)

__all__ = ["train_bnn"]


def _create_train_state(rng, model, optimizer, **init_data):
    init_key, sample_key = jr.split(rng)
    params = model.init({"sample": sample_key, "params": init_key}, **init_data)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    return state


# pylint: disable=too-many-locals
def train_bnn(
    rng_key,
    bnn: BNN,
    x: Array,
    y: Array,
    optimizer=optax.adam(3e-4),
    n_iter=10000,
    batch_size=128,
    verbose=False,
):
    r"""Train a Bayesian neural network.

    Parameters
    ----------
    rng_key: jax.random.PRNGKey
        a key for seeding random number generators
    bnn: BNN
        a GP object
    x: jax.Array
        an input array of dimension :math:`n \times p`
    y: an array of outputs of dimension :math:`n \times q`
    optimizer: optax.GradientTransformation
        an optax optimizer
    n_iter: int
        number of training iterations
    batch_size: int
        batch_size to
    verbose: bool
        print training details

    Returns
    -------
    Tuple[dict, jax.Array]
        a tuple of training parameters and training losses
    """
    itr_key, seed = jr.split(rng_key)
    # ignore this 'error', because mypy doesn't realize that this is correct
    train_itr = as_batch_iterator(
        itr_key,
        namedtuple("data", "y x")(y, x),  # type: ignore
        batch_size,
        True,
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
