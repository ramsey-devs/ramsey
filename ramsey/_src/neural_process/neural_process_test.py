# pylint: skip-file

import chex
import pytest
from jax import random as jr

from ramsey._src.data.data import sample_from_gaussian_process
from ramsey._src.neural_process.neural_process import NP
from ramsey._src.nn.MLP import MLP


#  pylint: disable=too-many-locals,invalid-name,redefined-outer-name
def test_module_dimensionality():
    key = jr.PRNGKey(1)
    data = sample_from_gaussian_process(key)
    x_target, y_target = data.x, data.y

    def module():
        np = NP(
            decoder=MLP([3, 2], name="decoder"),
            deterministic_encoder=MLP([4, 4], name="deterministic_encoder"),
            latent_encoder=(
                MLP([3, 3], name="latent_encoder1"),
                MLP([3, 6], name="latent_encoder2"),
            ),
        )
        return np

    f = module()
    params = f.init(
        {"sample": key, "params": key},
        x_context=x_target[:, :10, :],
        y_context=y_target[:, :10, :],
        x_target=x_target,
    )

    params = params["params"]
    chex.assert_shape(params["latent_encoder_0"]["linear_0"]["kernel"], (2, 3))
    chex.assert_shape(params["latent_encoder_0"]["linear_1"]["kernel"], (3, 3))
    chex.assert_shape(params["latent_encoder_1"]["linear_0"]["kernel"], (3, 3))
    chex.assert_shape(
        params["latent_encoder_1"]["linear_1"]["kernel"], (3, 2 * 3)
    )
    chex.assert_shape(
        params["deterministic_encoder"]["linear_0"]["kernel"], (2, 4)
    )
    chex.assert_shape(
        params["deterministic_encoder"]["linear_1"]["kernel"], (4, 4)
    )
    chex.assert_shape(params["decoder"]["linear_0"]["kernel"], (3 + 4 + 1, 3))
    chex.assert_shape(params["decoder"]["linear_1"]["kernel"], (3, 2))


def test_modules(module):
    key = jr.PRNGKey(1)
    data = sample_from_gaussian_process(key)
    x_target, y_target = data.x, data.y

    f = module()
    params = f.init(
        {"sample": key, "params": key},
        x_context=x_target[:, :10, :],
        y_context=y_target[:, :10, :],
        x_target=x_target,
    )
    y_star = f.apply(
        variables=params,
        rngs={"sample": key},
        x_context=x_target[:, :10, :],
        y_context=y_target[:, :10, :],
        x_target=x_target,
    )
    chex.assert_equal_shape([y_target, y_star.mean])


def test_modules_false_decoder():
    def module():
        np = NP(
            decoder=MLP([3, 3], name="decoder"),
            deterministic_encoder=MLP([4, 4], name="deterministic_encoder"),
            latent_encoder=(
                MLP([3, 3], name="latent_encoder1"),
                MLP([3, 6], name="latent_encoder2"),
            ),
        )
        return np

    key = jr.PRNGKey(1)
    data = sample_from_gaussian_process(key)
    x_target, y_target = data.x, data.y

    with pytest.raises(ValueError):
        f = module()
        f.init(
            {"sample": key, "params": key},
            x_context=x_target[:, :10, :],
            y_context=y_target[:, :10, :],
            x_target=x_target,
        )
