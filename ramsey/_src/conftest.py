# pylint: skip-file

import pytest

from ramsey import ANP, DANP, NP
from ramsey.nn import MLP, MultiHeadAttention


def lnp():
  np = NP(
    decoder=MLP([3, 2]),
    latent_encoder=(MLP([3, 3]), MLP([3, 6])),
  )
  return np


def np():
  np = NP(
    decoder=MLP([3, 2]),
    deterministic_encoder=MLP([4, 4]),
    latent_encoder=(MLP([3, 3]), MLP([3, 6])),
  )
  return np


def anp():
  np = ANP(
    decoder=MLP([3, 2]),
    deterministic_encoder=(
      MLP([4, 4]),
      MultiHeadAttention(num_heads=8, embedding=MLP([8, 8])),
    ),
    latent_encoder=(MLP([3, 3]), MLP([3, 6])),
  )
  return np


def danp():
  np = DANP(
    decoder=MLP([3, 2]),
    deterministic_encoder=(
      MLP([4, 4]),
      MultiHeadAttention(num_heads=8, embedding=MLP([8, 8])),
      MultiHeadAttention(num_heads=8, embedding=MLP([8, 8])),
    ),
    latent_encoder=(
      MLP([3, 3]),
      MultiHeadAttention(num_heads=8, embedding=MLP([8, 8])),
      MLP([3, 6]),
    ),
  )
  return np


@pytest.fixture(params=[lnp, np, anp, danp])
def module(request):
  yield request.param
