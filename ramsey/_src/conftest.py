# pylint: skip-file

import haiku as hk
import pytest

from ramsey import ANP, DANP, NP
from ramsey.attention import MultiHeadAttention


def __lnp(**kwargs):
    np = NP(
        decoder=hk.nets.MLP([3, 2]),
        latent_encoder=(hk.nets.MLP([3, 3]), hk.nets.MLP([3, 6])),
    )
    return np(**kwargs)


def __np(**kwargs):
    np = NP(
        decoder=hk.nets.MLP([3, 2]),
        deterministic_encoder=hk.nets.MLP([4, 4]),
        latent_encoder=(hk.nets.MLP([3, 3]), hk.nets.MLP([3, 6])),
    )
    return np(**kwargs)


def __anp(**kwargs):
    np = ANP(
        decoder=hk.nets.MLP([3, 2]),
        deterministic_encoder=(
            hk.nets.MLP([4, 4]),
            MultiHeadAttention(8, 8, hk.nets.MLP([8, 8])),
        ),
        latent_encoder=(hk.nets.MLP([3, 3]), hk.nets.MLP([3, 6])),
    )
    return np(**kwargs)


def __danp(**kwargs):
    np = DANP(
        decoder=hk.nets.MLP([3, 2]),
        deterministic_encoder=(
            hk.nets.MLP([4, 4]),
            MultiHeadAttention(8, 8, hk.nets.MLP([8, 8])),
            MultiHeadAttention(8, 8, hk.nets.MLP([8, 8])),
        ),
        latent_encoder=(
            hk.nets.MLP([3, 3]),
            MultiHeadAttention(8, 8, hk.nets.MLP([8, 8])),
            hk.nets.MLP([3, 6]),
        ),
    )
    return np(**kwargs)


@pytest.fixture(params=[__lnp, __np, __anp, __danp])
def module(request):
    yield request.param
