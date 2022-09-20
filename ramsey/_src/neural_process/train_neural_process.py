import haiku as hk
import jax
import optax
from haiku import Transformed
from haiku._src.data_structures import FlatMapping
from jax import numpy as np
from jax import random


# pylint: disable=too-many-locals
def train_neural_process(
    fn: Transformed,  # pylint: disable=invalid-name
    params: FlatMapping,
    rng: random.PRNGKey,
    x: np.ndarray,  # pylint: disable=invalid-name
    y: np.ndarray,  # pylint: disable=invalid-name
    n_context: int,
    n_target: int,
    n_iter=20000,
    stepsize=3e-4,
):
    if n_target < n_context:
        raise ValueError("'n_target' should be larger than 'n_context'")

    @jax.jit
    def _objective(par, key, x_context, y_context, x_target, y_target):
        _, obj = fn.apply(
            params=par,
            rng=key,
            x_context=x_context,
            y_context=y_context,
            x_target=x_target,
            y_target=y_target,
        )
        return obj

    @jax.jit
    def step(params, opt_state, rng, x_context, y_context, x_target, y_target):
        loss, grads = jax.value_and_grad(_objective)(
            params, rng, x_context, y_context, x_target, y_target
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    rng_seq = hk.PRNGSequence(rng)
    optimizer = optax.adam(stepsize)
    opt_state = optimizer.init(params)

    objectives = [0.0] * n_iter
    for _ in range(n_iter):
        x_context, y_context, x_target, y_target = _split_data(
            next(rng_seq), x, y, n_context, n_target
        )
        params, opt_state, loss_value = step(
            params,
            opt_state,
            next(rng_seq),
            x_context,
            y_context,
            x_target,
            y_target,
        )
        objectives[_] = loss_value
        if _ % 1000 == 0 or _ == n_iter - 1:
            elbo = -float(loss_value)
            print(f"ELBO at {_}: {elbo}")

    return params, objectives


def _split_data(
    key: random.PRNGKey,
    x: np.ndarray,  # pylint: disable=invalid-name
    y: np.ndarray,  # pylint: disable=invalid-name
    n_context: int,
    n_target: int,
):
    ibatch = random.choice(key, x.shape[0], shape=(2,), replace=False)
    idxs = random.choice(
        key, x.shape[1], shape=(n_context + n_target,), replace=False
    )
    x_context = x[ibatch][:, idxs[:n_context], :]
    y_context = y[ibatch][:, idxs[:n_context], :]
    x_target = x[ibatch][:, idxs, :]
    y_target = y[ibatch][:, idxs, :]

    return x_context, y_context, x_target, y_target
