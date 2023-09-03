import jax
import numpy as np
import optax
from flax.training.train_state import TrainState
from jax import Array
from jax import random as jr
from tqdm import tqdm

from ramsey._src.neural_process.neural_process import NP


@jax.jit
def step(rngs, state, **batch):
    current_step = state.step
    rngs = {name: jr.fold_in(rng, current_step) for name, rng in rngs.items()}

    def obj_fn(params):
        _, obj = state.apply_fn(variables=params, rngs=rngs, **batch)
        return obj

    obj, grads = jax.value_and_grad(obj_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, obj


# pylint: disable=too-many-locals
def train_neural_process(
    seed: jr.PRNGKey,
    neural_process: NP,  # pylint: disable=invalid-name
    x: Array,  # pylint: disable=invalid-name
    y: Array,  # pylint: disable=invalid-name
    n_context: int,
    n_target: int,
    optimizer=optax.adam(3e-4),
    n_iter=20000,
    verbose=False,
):
    rng, seed = jr.split(seed)
    state = create_train_state(
        rng,
        neural_process,
        optimizer,
        x_context=x,
        y_context=y,
        x_target=x,
    )

    objectives = np.zeros(n_iter)
    for i in tqdm(range(n_iter)):
        split_rng_key, sample_rng_key, seed = jr.split(seed, 3)
        batch = _split_data(split_rng_key, x, y, n_context, n_target)
        state, obj = step({"sample": sample_rng_key}, state, **batch)
        objectives[i] = obj
        if (i % 100 == 0 or i == n_iter - 1) and verbose:
            elbo = -float(obj)
            print(f"ELBO at itr {i}: {elbo:.2f}")

    return state.params, objectives


# pylint: disable=too-many-locals
def _split_data(
    seed: jr.PRNGKey,
    x: Array,  # pylint: disable=invalid-name
    y: Array,  # pylint: disable=invalid-name
    n_context: int,
    n_target: int,
):
    batch_rng_key, idx_rng_key, seed = jr.split(seed, 3)
    # TODO(s): why are we only taking two batches??
    ibatch = jr.choice(batch_rng_key, x.shape[0], shape=(2,), replace=False)
    idxs = jr.choice(
        idx_rng_key, x.shape[1], shape=(n_context + n_target,), replace=False
    )
    x_context = x[ibatch][:, idxs[:n_context], :]
    y_context = y[ibatch][:, idxs[:n_context], :]
    x_target = x[ibatch][:, idxs, :]
    y_target = y[ibatch][:, idxs, :]

    return {
        "x_context": x_context,
        "y_context": y_context,
        "x_target": x_target,
        "y_target": y_target,
    }


def create_train_state(rng, model, optimizer, **init_data):
    init_key, sample_key = jr.split(rng)
    params = model.init({"sample": sample_key, "params": init_key}, **init_data)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    return state
