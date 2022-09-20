import jax
import optax
from haiku import Transformed
from haiku._src.data_structures import FlatMapping
from jax import numpy as np


# pylint: disable=too-many-locals
def train_deepar(
    fn: Transformed,
    params: FlatMapping,
    x: np.ndarray,
    y: np.ndarray,
    n_iter=20000,
    stepsize=3e-4,
):
    @jax.jit
    def _objective(par, x, y):
        _, obj = fn.apply(
            params=par,
            x=x,
            y=y,
        )
        return obj

    @jax.jit
    def step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(_objective)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    optimizer = optax.adam(stepsize)
    opt_state = optimizer.init(params)

    objectives = [0.0] * n_iter
    for _ in range(n_iter):
        params, opt_state, loss_value = step(
            params,
            opt_state,
            x,
            y,
        )
        objectives[_] = loss_value
        if _ % 1000 == 0 or _ == n_iter - 1:
            nll = -float(loss_value)
            print(f"NLL at {_}: {nll}")

    return params, objectives
