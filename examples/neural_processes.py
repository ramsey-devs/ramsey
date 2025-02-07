# ruff: noqa: D103,PLR0913
"""Attentive neural process example.

Here, we implement and train an attentive neural process
and visualize predictions thereof.

References:
  Kim, Hyunjik, et al. "Attentive Neural Processes."
    International Conference on Learning Representations. 2019.
"""

import argparse

import matplotlib.pyplot as plt
from flax import nnx
from jax import numpy as jnp
from jax import random as jr

from ramsey import ANP, DANP, NP, train_neural_process
from ramsey.data import sample_from_gaussian_process
from ramsey.nn import MLP, MultiHeadAttention


def data(key):
  data = sample_from_gaussian_process(key, batch_size=10, num_observations=200)
  return (data.x, data.y), data.f


def get_neural_process(method):
  x_dim = y_dim = 1
  # any dimensionality
  dim = 128
  # we concat x and y
  in_features = x_dim + y_dim
  # in features latent is same as dim
  in_features_latent = dim
  # 2 * latent dim to parameterize Gaussian
  out_features_latent = dim * 2
  # just the dim
  out_features_det = dim
  # latent + deterministic + y dims
  in_features_decoder = out_features_latent // 2 + out_features_det + 1
  # dimensionality of the family, e.g., Gaussian has two parameters, so 2
  out_features_decoder = 2
  if method == "NP":
    model = NP(
      latent_encoder=(
        MLP(in_features, [dim, in_features_latent], rngs=nnx.Rngs(0)),
        MLP(
          in_features_latent,
          [dim, out_features_latent],
          rngs=nnx.Rngs(1),
        ),
      ),
      deterministic_encoder=MLP(
        in_features, [dim, out_features_det], rngs=nnx.Rngs(2)
      ),
      decoder=MLP(
        in_features_decoder, [dim, dim, out_features_decoder], rngs=nnx.Rngs(3)
      ),
      rngs=nnx.Rngs(4),
    )
  elif method == "ANP":
    model = ANP(
      latent_encoder=(
        MLP(in_features, [dim, in_features_latent], rngs=nnx.Rngs(0)),
        MLP(in_features_latent, [dim, out_features_latent], rngs=nnx.Rngs(1)),
      ),
      deterministic_encoder=(
        MLP(in_features, [dim, out_features_det], rngs=nnx.Rngs(2)),
        MultiHeadAttention(
          in_features=out_features_det,
          num_heads=4,
          embedding=MLP(x_dim, [dim, out_features_det], rngs=nnx.Rngs(3)),
          rngs=nnx.Rngs(4),
        ),
      ),
      decoder=MLP(
        in_features_decoder, [dim, dim, out_features_decoder], rngs=nnx.Rngs(5)
      ),
      rngs=nnx.Rngs(6),
    )
  elif method == "DANP":
    model = DANP(
      latent_encoder=(
        MLP(in_features, [dim, in_features_latent], rngs=nnx.Rngs(0)),
        MultiHeadAttention(
          in_features=in_features_latent,
          num_heads=4,
          embedding=lambda x: x,  # not needed since dims fit
          rngs=nnx.Rngs(1),
        ),
        MLP(in_features_latent, [dim, out_features_latent], rngs=nnx.Rngs(2)),
      ),
      deterministic_encoder=(
        MLP(in_features, [dim, out_features_det], rngs=nnx.Rngs(3)),
        MultiHeadAttention(
          in_features=out_features_det,
          num_heads=4,
          embedding=lambda x: x,  # not needed since dims fit
          rngs=nnx.Rngs(4),
        ),
        MultiHeadAttention(
          in_features=out_features_det,
          num_heads=4,
          embedding=MLP(x_dim, [dim, out_features_det], rngs=nnx.Rngs(5)),
          rngs=nnx.Rngs(6),
        ),
      ),
      decoder=MLP(
        in_features_decoder, [dim, dim, out_features_decoder], rngs=nnx.Rngs(7)
      ),
      rngs=nnx.Rngs(8),
    )
  else:
    raise ValueError("model incorrectly specified")
  return model


def train_np(
  rng_key, n_context, n_target, x_target, y_target, method, num_iter
):
  neural_process = get_neural_process(method)
  neural_process = train_neural_process(
    rng_key,
    neural_process,
    x=x_target,
    y=y_target,
    n_context=n_context,
    n_target=n_target,
    n_iter=num_iter,
    batch_size=2,
  )
  return neural_process


def plot(
  seed,
  neural_process,
  x_target,
  y_target,
  f_target,
  n_context,
  n_target,
):
  sample_key, seed = jr.split(seed)
  sample_idxs = jr.choice(
    sample_key,
    x_target.shape[1],
    shape=(n_context + n_target,),
    replace=False,
  )

  idxs = [0, 2, 5, 7]
  _, axes = plt.subplots(figsize=(10, 6), nrows=2, ncols=2)
  for _, (idx, ax) in enumerate(zip(idxs, axes.flatten())):
    x = jnp.squeeze(x_target[idx, :, :])
    f = jnp.squeeze(f_target[idx, :, :])
    y = jnp.squeeze(y_target[idx, :, :])

    srt_idxs = jnp.argsort(x)
    ax.plot(x[srt_idxs], f[srt_idxs], color="blue", alpha=0.75)
    ax.scatter(
      x[sample_idxs[:n_context]],
      y[sample_idxs[:n_context]],
      color="blue",
      marker="+",
      alpha=0.75,
    )

    neural_process.eval()
    for _ in range(20):
      sample_rng_key, seed = jr.split(seed, 2)
      y_star = neural_process(
        x_context=x[jnp.newaxis, sample_idxs, jnp.newaxis],
        y_context=y[jnp.newaxis, sample_idxs, jnp.newaxis],
        x_target=x_target[[idx], :, :],
      ).mean
      x_star = jnp.squeeze(x_target[[idx], :, :])
      y_star = jnp.squeeze(y_star)
      ax.plot(x_star[srt_idxs], y_star[srt_idxs], color="black", alpha=0.1)
    ax.grid()
    ax.set_frame_on(False)
  plt.show()


def run(args):
  n_context, n_target = (5, 10), (20, 30)
  data_rng_key, train_rng_key, plot_rng_key = jr.split(jr.PRNGKey(0), 3)
  (x_target, y_target), f_target = data(data_rng_key)

  neural_process = train_np(
    train_rng_key,
    n_context,
    n_target,
    x_target,
    y_target,
    args.method,
    args.num_iter,
  )
  plot(
    plot_rng_key,
    neural_process,
    x_target,
    y_target,
    f_target,
    n_context=10,
    n_target=20,
  )


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-m", "--method", type=str, default="DANP")
  parser.add_argument("-n", "--num_iter", type=int, default=10_000)
  run(parser.parse_args())
