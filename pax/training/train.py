import jax
from jax import random
import optax

from haiku import Transformed
from haiku._src.data_structures import FlatMapping


@jax.jit
def update(objective, optimizer, params, opt_state):
    val, grads = jax.value_and_grad(objective)(params)
    updates, new_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_state, val


def train(
    fn: Transformed,
    params: FlatMapping,
    rng=random.PRNGKey,
    n_iter=10000,
    stepsize=0.001,
    **kwargs
):
    def _objective(**kwargs):
        _, obj = fn.apply(rng=rng, **kwargs)
        return obj

    optimizer = optax.adam(stepsize)
    opt_state = optimizer.init(params)

    objectives = []
    for step in range(n_iter):
        params, opt_state, val = update(
            _objective, optimizer, params, opt_state
        )
        objectives.append(float(val))

    return params, objectives
