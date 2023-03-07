# pylint: skip-file

import chex
import haiku as hk
import pytest
from jax import random

from ramsey import NP
from ramsey.data import sample_from_gaussian_process


#  pylint: disable=too-many-locals,invalid-name,redefined-outer-name
def test_module_dimensionality():
    key = random.PRNGKey(1)
    (x_target, y_target), _ = sample_from_gaussian_process(key)

    def module(**kwargs):
        np = NP(
            decoder=hk.nets.MLP([3, 2], name="decoder"),
            deterministic_encoder=hk.nets.MLP(
                [4, 4], name="deterministic_encoder"
            ),
            latent_encoder=(
                hk.nets.MLP([3, 3], name="latent_encoder1"),
                hk.nets.MLP([3, 6], name="latent_encoder2"),
            ),
        )
        return np(**kwargs)

    f = hk.transform(module)
    params = f.init(
        key,
        x_context=x_target[:, :10, :],
        y_context=y_target[:, :10, :],
        x_target=x_target,
    )

    chex.assert_shape(params["latent_encoder1/~/linear_0"]["w"], (2, 3))
    chex.assert_shape(params["latent_encoder1/~/linear_1"]["w"], (3, 3))
    chex.assert_shape(params["latent_encoder2/~/linear_0"]["w"], (3, 3))
    chex.assert_shape(params["latent_encoder2/~/linear_1"]["w"], (3, 2 * 3))
    chex.assert_shape(params["deterministic_encoder/~/linear_0"]["w"], (2, 4))
    chex.assert_shape(params["deterministic_encoder/~/linear_1"]["w"], (4, 4))
    chex.assert_shape(params["decoder/~/linear_0"]["w"], (3 + 4 + 1, 3))
    chex.assert_shape(params["decoder/~/linear_1"]["w"], (3, 2))


def test_modules(module):
    key = random.PRNGKey(1)
    (x_target, y_target), _ = sample_from_gaussian_process(key)

    f = hk.transform(module)
    params = f.init(
        key,
        x_context=x_target[:, :10, :],
        y_context=y_target[:, :10, :],
        x_target=x_target,
    )
    y_star = f.apply(
        rng=key,
        params=params,
        x_context=x_target[:, :10, :],
        y_context=y_target[:, :10, :],
        x_target=x_target,
    )
    chex.assert_equal_shape([y_target, y_star.mean])


def test_modules_false_decoder():
    def f(**kwargs):
        np = NP(
            decoder=hk.nets.MLP([3, 3], name="decoder"),
            deterministic_encoder=hk.nets.MLP(
                [4, 4], name="deterministic_encoder"
            ),
            latent_encoder=(
                hk.nets.MLP([3, 3], name="latent_encoder1"),
                hk.nets.MLP([3, 6], name="latent_encoder2"),
            ),
        )
        return np(**kwargs)

    key = random.PRNGKey(1)
    (x_target, y_target), _ = sample_from_gaussian_process(key)

    with pytest.raises(ValueError):
        f = hk.transform(f)
        f.init(
            key,
            x_context=x_target[:, :10, :],
            y_context=y_target[:, :10, :],
            x_target=x_target,
        )
