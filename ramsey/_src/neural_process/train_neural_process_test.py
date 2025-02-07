
from jax import random as jr

from ramsey import train_neural_process
from ramsey.data import sample_from_gaussian_process


def test_neural_process_training(module):
    key = jr.PRNGKey(1)
    data = sample_from_gaussian_process(key)
    x_target, y_target = data.x, data.y

    key, train_key = jr.split(key)
    train_neural_process(
        train_key,
        module(),
        x=x_target,
        y=y_target,
        n_context=10,
        n_target=20,
        n_iter=10,
        batch_size=2,
    )
