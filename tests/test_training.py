import haiku as hk
from jax import random

from ramsey._src.train import train_neural_process


#  pylint: disable=too-many-locals,invalid-name,redefined-outer-name
def test_neural_process_training(simple_data_set, module):
    key = random.PRNGKey(1)
    _, _, x_target, y_target = simple_data_set

    f = hk.transform(module)
    params = f.init(
        key, x_context=x_target, y_context=y_target, x_target=x_target
    )

    key, train_key = random.split(key)
    train_neural_process(
        f,
        params,
        train_key,
        n_iter=100,
        x=x_target,
        y=y_target,
        n_context=10,
        n_target=10,
    )
