# pylint: skip-file

import chex
import pytest
from flax import nnx
from jax import random as jr

from ramsey._src.data.data import sample_from_gaussian_process
from ramsey._src.neural_process.neural_process import NP
from ramsey._src.nn.MLP import MLP


def test_modules(module):
    key = jr.PRNGKey(1)
    data = sample_from_gaussian_process(key)
    x_target, y_target = data.x, data.y

    f = module()
    y_star = f(
        x_context=x_target[:, :10, :],
        y_context=y_target[:, :10, :],
        x_target=x_target,
    )
    chex.assert_equal_shape([y_target, y_star.mean])


def test_modules_false_decoder():
    def module():
        np = NP(
            decoder=MLP(1 + 3 + 3, [3, 3], rngs=nnx.Rngs(0)),
            deterministic_encoder=MLP(1, [3, 3], rngs=nnx.Rngs(0)),
            latent_encoder=(
                MLP(1, [3, 3], rngs=nnx.Rngs(0)),
                MLP(3, [3, 6], rngs=nnx.Rngs(0)),
            ),
        )
        return np

    key = jr.PRNGKey(1)
    data = sample_from_gaussian_process(key)
    x_target, y_target = data.x, data.y

    with pytest.raises(ValueError):
        f = module()
        f(
            x_context=x_target[:, :10, :],
            y_context=y_target[:, :10, :],
            x_target=x_target,
        )
