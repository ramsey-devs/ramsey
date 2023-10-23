import jax
import numpy as np
import optax
from flax.training.train_state import TrainState
from jax import Array
from jax import numpy as jnp
from jax import random as jr
from tqdm import tqdm

from ramsey._src.experimental.gaussian_process.gaussian_process import GP
from ramsey._src.experimental.gaussian_process.sparse_gaussian_process import (
    SparseGP,
)


# pylint: disable=too-many-locals,invalid-name
def train_gaussian_process(
    rng_key: jr.PRNGKey,
    gaussian_process: GP,
    x: Array,
    y: Array,
    optimizer=optax.adam(3e-3),
    n_iter=1000,
    verbose=False,
):
    r"""Train a Gaussian process.

    Parameters
    ----------
    rng_key: jax.random.PRNGKey
        a key for seeding random number generators
    gaussian_process: GP
        a GP object
    x: jax.Array
        an input array of dimension :math:`n \times p`
    y: an array of outputs of dimension :math:`n \times 1`
    optimizer: optax.GradientTransformation
        an optax optimizer
    n_iter: int
        number of training iterations
    verbose: bool
        print training details

    Returns
    -------
    Tuple[dict, jax.Array]
        a tuple of training parameters and training losses
    """

    @jax.jit
    def step(rngs, state, **batch):
        step = state.step
        rngs = {name: jr.fold_in(rng, step) for name, rng in rngs.items()}

        def obj_fn(params):
            mvn = state.apply_fn(variables=params, rngs=rngs, x=batch["x"])
            # TODO(simon): should not return mvn but the logprob
            mll = mvn.log_prob(batch["y"].T)
            return -jnp.sum(mll)

        obj, grads = jax.value_and_grad(obj_fn)(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, obj

    train_state_rng, rng_key = jr.split(rng_key)
    state = _create_train_state(
        train_state_rng, gaussian_process, optimizer, x=x
    )

    objectives = np.zeros(n_iter)
    for i in tqdm(range(n_iter)):
        sample_rng_key, rng_key = jr.split(rng_key)
        state, obj = step({"sample": sample_rng_key}, state, x=x, y=y)
        objectives[i] = obj
        if (i % 100 == 0 or i == n_iter - 1) and verbose:
            mll = -float(obj)
            print(f"MLL at itr {i}: {mll:.2f}")

    return state.params, objectives


# pylint: disable=too-many-locals,invalid-name
def train_sparse_gaussian_process(
    rng_key: jr.PRNGKey,
    gaussian_process: SparseGP,
    x: Array,
    y: Array,
    optimizer=optax.adam(3e-3),
    n_iter=1000,
    verbose=False,
):
    r"""Train a sparse Gaussian process.

    Parameters
    ----------
    rng_key: jax.random.PRNGKey
        a key for seeding random number generators
    gaussian_process: SparseGP
        a SparseGP object
    x: jax.Array
        an input array of dimension :math:`n \times p`
    y: an array of outputs of dimension :math:`n \times 1`
    optimizer: optax.GradientTransformation
        an optax optimizer
    n_iter: int
        number of training iterations
    verbose: bool
        print training details

    Returns
    -------
    Tuple[dict, jax.Array]
        a tuple of training parameters and training losses
    """

    @jax.jit
    def step(rngs, state, **batch):
        def obj_fn(params):
            elbo = state.apply_fn(variables=params, rngs=rngs, **batch)
            return -elbo

        obj, grads = jax.value_and_grad(obj_fn)(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, obj

    train_state_key, rng_key = jr.split(rng_key)
    state = _create_train_state(
        train_state_key, gaussian_process, optimizer, x=x, y=y
    )

    objectives = np.zeros(n_iter)
    for i in tqdm(range(n_iter)):
        sample_rng_key, rng_key = jr.split(rng_key)
        state, obj = step({"sample": sample_rng_key}, state, x=x, y=y)
        objectives[i] = obj
        if (i % 100 == 0 or i == n_iter - 1) and verbose:
            elbo = -float(obj)
            print(f"ELBO at itr {i}: {elbo:.2f}")

    return state.params, objectives


def _create_train_state(rng, model, optimizer, **init_data):
    params = model.init({"params": rng}, **init_data)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    return state
