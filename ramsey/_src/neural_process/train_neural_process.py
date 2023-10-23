import jax
import numpy as np
import optax
from flax.training.train_state import TrainState
from jax import Array
from jax import random as jr
from tqdm import tqdm

from ramsey._src.neural_process.neural_process import NP

__all__ = ["train_neural_process"]


@jax.jit
def _step(rngs, state, **batch):
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
    rng_key: jr.PRNGKey,
    neural_process: NP,  # pylint: disable=invalid-name
    x: Array,  # pylint: disable=invalid-name
    y: Array,  # pylint: disable=invalid-name
    n_context: int,
    n_target: int,
    batch_size: int,
    optimizer=optax.adam(3e-4),
    n_iter=20000,
    verbose=False,
):
    r"""Train a neural process.

    Utility function to train a latent or conditional neural process, i.e.,
    a process belonging to the `NP` class.

    Parameters
    ----------
    rng_key: jax.random.PRNGKey
        a key for seeding random number generators
    neural_process: Union[NP, ANP, DANP]
        an object that inherits from NP
    x: jax.Array
        array of inputs. Should be a tensor of dimension
        :math:`b \times n \times p`
        where :math:`b` indexes a sequence of batches, e.g., different time
        series, :math:`n` indexes the number of observations per batch, e.g.,
        time points, and :math:`p` indexes the number of feats
    y: jax.Array
        array of outputs. Should be a tensor of dimension
        :math:`b \times n \times q`
        where :math:`b` and :math:`n` are the same as for :math:`x` and
        :math:`q` is the number of outputs
    n_context: int
        number of context points
    n_target: int
        number of target points
    batch_size: int
        number of elements that are samples for each gradient step, i.e.,
        number of elements in first axis of :math:`x` and :math:`y`
    optimizer: optax.GradientTransformation
        an optax optimizer object
    n_iter: int
        number of training iterations
    verbose: bool
        true if print training progress

    Returns
    -------
    Tuple[dict, jnp.Array]
        returns a tuple of trained parameters and training loss profile
    """
    train_state_rng, rng_key = jr.split(rng_key)
    state = _create_train_state(
        train_state_rng,
        neural_process,
        optimizer,
        x_context=x,
        y_context=y,
        x_target=x,
    )

    objectives = np.zeros(n_iter)
    for i in tqdm(range(n_iter)):
        split_rng_key, sample_rng_key, rng_key = jr.split(rng_key, 3)
        batch = _split_data(
            split_rng_key,
            x,
            y,
            n_context=n_context,
            n_target=n_target,
            batch_size=batch_size,
        )
        state, obj = _step({"sample": sample_rng_key}, state, **batch)
        objectives[i] = obj
        if (i % 100 == 0 or i == n_iter - 1) and verbose:
            elbo = -float(obj)
            print(f"ELBO at itr {i}: {elbo:.2f}")

    return state.params, objectives


# pylint: disable=too-many-locals
def _split_data(
    rng_key: jr.PRNGKey,
    x: Array,  # pylint: disable=invalid-name
    y: Array,  # pylint: disable=invalid-name
    batch_size: int,
    n_context: int,
    n_target: int,
):
    batch_rng_key, idx_rng_key, rng_key = jr.split(rng_key, 3)
    ibatch = jr.choice(
        batch_rng_key, x.shape[0], shape=(batch_size,), replace=False
    )
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


def _create_train_state(rng, model, optimizer, **init_data):
    init_key, sample_key = jr.split(rng)
    params = model.init({"sample": sample_key, "params": init_key}, **init_data)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    return state
