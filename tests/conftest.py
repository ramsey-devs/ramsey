import haiku as hk
import pytest
from jax import numpy as np
from jax import random
from numpyro import distributions as dist

from ramsey.attention import MultiHeadAttention
from ramsey.covariance_functions import exponentiated_quadratic
from ramsey.models import ANP, DANP, NP


#  pylint: disable=too-many-locals,invalid-name,redefined-outer-name
@pytest.fixture()
def simple_data_set():
    key = random.PRNGKey(0)
    batch_size = 10
    n, p = 50, 1
    n_context = 20

    key, sample_key = random.split(key, 2)
    x = random.normal(key, shape=(n * p,)).reshape((n, p))
    ys = []
    for _ in range(batch_size):
        key, sample_key1, sample_key2, sample_key3 = random.split(key, 4)
        rho = dist.InverseGamma(5, 5).sample(sample_key1)
        sigma = dist.InverseGamma(5, 5).sample(sample_key2)
        K = exponentiated_quadratic(x, x, sigma, rho)
        y = random.multivariate_normal(
            sample_key3, mean=np.zeros(n), cov=K + np.diag(np.ones(n)) * 0.05
        ).reshape((1, n, 1))
        ys.append(y)

    x_target = np.tile(x, [batch_size, 1, 1])
    y_target = np.vstack(np.array(ys))

    key, sample_key = random.split(key, 2)
    idxs_context = random.choice(
        sample_key, np.arange(n), shape=(n_context,), replace=False
    )

    x_context = x_target[:, idxs_context, :]
    y_context = y_target[:, idxs_context, :]

    return x_context, y_context, x_target, y_target


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
