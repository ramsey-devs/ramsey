import jax
import optax
from haiku import Transformed
from haiku._src.data_structures import FlatMapping
from jax import random


def train(
    fn: Transformed,  # pylint: disable=invalid-name
    params: FlatMapping,
    rng=random.PRNGKey,
    n_iter=10000,
    stepsize=0.001,
    **kwargs,
):
    def _objective(par):
        _, obj = fn.apply(params=par, rng=rng, **kwargs)
        return obj

    @jax.jit
    def _update(par, state):
        val, grads = jax.value_and_grad(_objective)(par)
        updates, new_state = optimizer.update(grads, state)
        new_params = optax.apply_updates(par, updates)
        return new_params, new_state, val

    optimizer = optax.adam(stepsize)
    opt_state = optimizer.init(params)

    objectives = [0.0] * n_iter
    for step in range(n_iter):
        params, opt_state, objective = _update(params, opt_state)
        objectives[step] = float(objective)

    return params, objectives
