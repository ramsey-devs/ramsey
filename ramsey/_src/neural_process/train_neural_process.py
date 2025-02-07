import jax
import numpy as np
import optax
from flax.training.train_state import TrainState
from jax import random as jr
from tqdm import tqdm

from ramsey import ANP, DANP
from ramsey._src.neural_process.neural_process import NP

__all__ = ["train_neural_process"]


@jax.jit
def _step(rngs, state, **batch):
  current_step = state.step
  rngs = {name: jr.fold_in(rng, current_step) for name, rng in rngs.items()}

  def obj_fn(params):
    return state.apply_fn(variables=params, rngs=rngs, method="loss", **batch)

  obj, grads = jax.value_and_grad(obj_fn)(state.params)
  new_state = state.apply_gradients(grads=grads)
  return new_state, obj


# pylint: disable=too-many-locals
def train_neural_process(
  rng_key: jr.key,
  neural_process: NP | ANP | DANP,
  x: jax.Array,
  y: jax.Array,
  n_context: int | tuple[int, int],
  n_target: int | tuple[int, int],
  batch_size: int,
  optimizer=optax.adam(3e-4),
  n_iter=20_000,
  verbose=False,
):
  r"""Train a neural process.

  Utility function to train a latent or conditional neural process, i.e.,
  a process belonging to the `NP` class.

  Args:
    rng_key: a key for seeding random number generators
    neural_process: an object that inherits from NP
    x: array of inputs. Should be a tensor of dimension
      :math:`b \times n \times p` where :math:`b` indexes a sequence of
      batches, e.g., different time series, :math:`n` indexes the number of
      observations per batch, e.g., time points, and :math:`p` indexes the
      number of features
    y: array of outputs. Should be a tensor of dimension
      :math:`b \times n \times q` where :math:`b` and :math:`n` are the same as
      for :math:`x` and  :math:`q` is the number of outputs
    n_context: number of context points. If a tuple is given samples the number
      of context points per iteration on the interval defined by the tuple.
    n_target: number of target points. If a tuple is given samples the number
      of context points per iteration on the interval defined by the tuple.
      The number of target points includes the
      number of context points, that means, if n_context=5 and n_target=10
      then the target set contains 5 more points than the context set but
      includes the contexts, too.
    batch_size: number of elements that are samples for each gradient step,
      i.e., number of elements in first axis of :math:`x` and :math:`y`
    optimizer: an optax optimizer object
    n_iter: number of training iterations
    verbose: true if print training progress

  Returns:
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


def _split_data(
  rng_key: jr.key,
  x: jax.Array,
  y: jax.Array,
  batch_size: int,
  n_context: int | tuple[int, int],
  n_target: int | tuple[int, int],
):
  if isinstance(n_context, tuple):
    cnt_key, rng_key = jr.split(rng_key)
    n_context = jr.randint(
      cnt_key, minval=n_context[0], maxval=n_context[1], shape=()
    )
  if isinstance(n_target, tuple):
    trg_key, rng_key = jr.split(rng_key)
    n_target = jr.randint(
      trg_key, minval=n_target[0], maxval=n_target[1], shape=()
    )

  batch_rng_key, idx_rng_key, rng_key = jr.split(rng_key, 3)  # type: ignore[operator]
  ibatch = jr.choice(
    batch_rng_key, x.shape[0], shape=(batch_size,), replace=False
  )
  idxs = jr.choice(
    idx_rng_key,
    x.shape[1],
    shape=(n_target,),
    replace=False,
  )
  ibatch = np.asarray(ibatch, dtype=np.int32)
  x_context = x[ibatch][:, idxs[:n_context], :]  # type: ignore[misc]
  y_context = y[ibatch][:, idxs[:n_context], :]  # type: ignore[misc]
  x_target = x[ibatch][:, idxs, :]
  y_target = y[ibatch][:, idxs, :]

  return {
    "x_context": x_context,
    "y_context": y_context,
    "x_target": x_target,
    "y_target": y_target,
  }


# ruff: noqa: ANN001,ANN003,PLR0913
def _create_train_state(rng, model, optimizer, **init_data):
  init_key, sample_key = jr.split(rng)
  params = model.init({"sample": sample_key, "params": init_key}, **init_data)
  state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
  return state
