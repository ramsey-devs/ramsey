# pylint: skip-file

import pytest
from flax import nnx

from ramsey import ANP, DANP, NP
from ramsey.nn import MLP, MultiHeadAttention


def lnp():
  np = NP(
    latent_encoder=(
      MLP(2, [3, 3], rngs=nnx.Rngs(0)),
      MLP(3, [3, 6], rngs=nnx.Rngs(0)),
    ),
    decoder=MLP(1 + 3, [3, 2], rngs=nnx.Rngs(0)),
    rngs=nnx.Rngs(0),
  )
  return np


def np():
  np = NP(
    latent_encoder=(
      MLP(2, [3, 3], rngs=nnx.Rngs(0)),
      MLP(3, [3, 6], rngs=nnx.Rngs(0)),
    ),
    deterministic_encoder=MLP(2, [4, 4], rngs=nnx.Rngs(0)),
    decoder=MLP(1 + 4 + 3, [3, 2], rngs=nnx.Rngs(0)),
    rngs=nnx.Rngs(0),
  )
  return np


def anp():
  np = ANP(
    latent_encoder=(
      MLP(
        2,
        [3, 3],
        rngs=nnx.Rngs(0),
      ),
      MLP(
        3,
        [3, 6],
        rngs=nnx.Rngs(0),
      ),
    ),
    deterministic_encoder=(
      MLP(2, [4, 4], rngs=nnx.Rngs(0)),
      MultiHeadAttention(
        4,
        num_heads=4,
        embedding=MLP(
          1,
          [3, 4],
          rngs=nnx.Rngs(0),
        ),
        rngs=nnx.Rngs(0),
      ),
    ),
    decoder=MLP(
      1 + 4 + 3,
      [3, 2],
      rngs=nnx.Rngs(0),
    ),
    rngs=nnx.Rngs(0),
  )
  return np


def danp():
  np = DANP(
    latent_encoder=(
      MLP(2, [3, 8], rngs=nnx.Rngs(0)),
      MultiHeadAttention(
        8,
        num_heads=4,
        embedding=MLP(8, [8, 8], rngs=nnx.Rngs(0)),
        rngs=nnx.Rngs(0),
      ),
      MLP(8, [3, 6], rngs=nnx.Rngs(0)),
    ),
    deterministic_encoder=(
      MLP(2, [4, 8], rngs=nnx.Rngs(0)),
      MultiHeadAttention(
        8,
        num_heads=4,
        embedding=MLP(8, [4, 8], rngs=nnx.Rngs(0)),
        rngs=nnx.Rngs(0),
      ),
      MultiHeadAttention(
        8,
        num_heads=4,
        embedding=MLP(1, [4, 8], rngs=nnx.Rngs(0)),
        rngs=nnx.Rngs(0),
      ),
    ),
    decoder=MLP(1 + 3 + 8, [3, 2], rngs=nnx.Rngs(0)),
    rngs=nnx.Rngs(0),
  )
  return np


@pytest.fixture(params=[lnp, np, anp, danp])
def module(request):
  yield request.param
