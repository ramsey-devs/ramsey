import haiku as hk
import jax
import optax
from haiku import Transformed
from haiku._src.data_structures import FlatMapping
from jax import numpy as jnp
from jax import random


# pylint: disable=too-many-locals
def train_gaussian_process(
    fn: Transformed,  # pylint: disable=invalid-name
    params: FlatMapping,
    rng: random.PRNGKey,
    x: jnp.ndarray,  # pylint: disable=invalid-name
    y: jnp.ndarray,  # pylint: disable=invalid-name
    n_iter=1000,
    stepsize=3e-03,
    verbose=False,
):
    def _objective(par, key, x, y):
        mvn = fn.apply(
            params=par,
            rng=key,
            x=x,
        )
        mll = mvn.log_prob(y.T)
        return -jnp.sum(mll)

    @jax.jit
    def step(params, opt_state, rng, x, y):
        loss, grads = jax.value_and_grad(_objective)(params, rng, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    rng_seq = hk.PRNGSequence(rng)
    optimizer = optax.adam(stepsize)
    opt_state = optimizer.init(params)

    objectives = [0.0] * n_iter
    for _ in range(n_iter):
        params, opt_state, loss_value = step(
            params, opt_state, next(rng_seq), x, y
        )
        objectives[_] = loss_value
        if (_ % 100 == 0 or _ == n_iter - 1) and verbose:
            mll = -float(loss_value)
            print(f"MLL at {_}: {mll:.2f}")

    return params, objectives


# pylint: disable=too-many-locals
def train_sparse_gaussian_process(
    fn: Transformed,  # pylint: disable=invalid-name
    params: FlatMapping,
    rng: random.PRNGKey,
    x: jnp.ndarray,  # pylint: disable=invalid-name
    y: jnp.ndarray,  # pylint: disable=invalid-name
    n_iter=1000,
    stepsize=3e-03,
):
    def _objective(par, key, x, y):
        variational_lower_bound = fn.apply(params=par, rng=key, x=x, y=y)
        return -variational_lower_bound

    @jax.jit
    def step(params, opt_state, rng, x, y):
        loss, grads = jax.value_and_grad(_objective)(params, rng, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    rng_seq = hk.PRNGSequence(rng)
    optimizer = optax.adam(stepsize)
    opt_state = optimizer.init(params)

    objectives = [0.0] * n_iter
    for _ in range(n_iter):
        params, opt_state, loss_value = step(
            params, opt_state, next(rng_seq), x=x, y=y
        )
        objectives[_] = loss_value
        if _ % 100 == 0 or _ == n_iter - 1:
            variational_lower_bound = -float(loss_value)
            print(f"Variational Lower Bound at {_}: {variational_lower_bound}")

    return params, objectives
