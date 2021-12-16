import jax
from jax import random
import optax

from haiku import Transformed
from haiku._src.data_structures import FlatMapping


def train(
    fn: Transformed,
    params: FlatMapping,
    rng=random.PRNGKey,
    n_iter=10000,
    stepsize=0.001,
    **kwargs
):
    def _objective(params):
        _, obj = fn.apply(params=params, rng=rng, **kwargs)
        return obj

    @jax.jit
    def _update(params, opt_state):
        val, grads = jax.value_and_grad(_objective)(params)
        updates, new_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, val

    optimizer = optax.adam(stepsize)
    opt_state = optimizer.init(params)

    objectives = [0.0] * n_iter
    for step in range(n_iter):
        params, opt_state, val = _update(params, opt_state)
        objectives[step] = float(val)

    return params, objectives
