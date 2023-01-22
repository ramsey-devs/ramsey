from typing import Callable

import haiku as hk
import jax
import optax
from haiku._src.data_structures import FlatMapping
from jax import numpy as jnp
from jax import random


# pylint: disable=too-many-locals
def train_model(
    model: hk.Transformed,  # pylint: disable=invalid-name
    objective: Callable,
    params: FlatMapping,
    rng: random.PRNGKey,
    x: jnp.ndarray,  # pylint: disable=invalid-name
    y: jnp.ndarray,  # pylint: disable=invalid-name
    n_iter=1000,
    stepsize=1e-3,
):
    @jax.jit
    def step(params, opt_state, rng, x, y):
        obj, grads = jax.value_and_grad(objective)(params, rng, model, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, obj

    rng_seq = hk.PRNGSequence(rng)
    optimizer = optax.adam(stepsize)
    opt_state = optimizer.init(params)
    objectives = [0.0] * n_iter

    for _ in range(n_iter):

        params, opt_state, loss = step(params, opt_state, next(rng_seq), x, y)

        objectives[_] = loss
        if _ % 200 == 0 or _ == n_iter - 1:
            print(f"step {_}: obj={loss:.5f}")

    objectives = jnp.asarray(objectives)

    return params, objectives
