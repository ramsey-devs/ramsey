import haiku as hk
import jax.random as random

from pax._src.train import train


def test_neural_process_training(simple_data_set, module):
    key = random.PRNGKey(1)
    x_context, y_context, x_target, y_target = simple_data_set

    f = hk.transform(module)
    params = f.init(
        key, x_context=x_context, y_context=y_context, x_target=x_target
    )

    key, train_key = random.split(key)
    train(
        f,
        params,
        train_key,
        n_iter=100,
        x_context=x_context,
        y_context=y_context,
        x_target=x_target,
        y_target=y_target,
    )
