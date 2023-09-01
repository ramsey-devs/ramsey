# pylint: skip-file

from jax import jr

from ramsey.data import sample_from_gaussian_process
from ramsey.train import train_neural_process


#  pylint: disable=too-many-locals,invalid-name,redefined-outer-name
def test_neural_process_training(module):
    key = jr.PRNGKey(1)
    (x_target, y_target), _ = sample_from_gaussian_process(key)

    params = module.init(
        key, x_context=x_target, y_context=y_target, x_target=x_target
    )

    key, train_key = jr.split(key)
    train_neural_process(
        module,
        params,
        train_key,
        n_iter=10,
        x=x_target,
        y=y_target,
        n_context=10,
        n_target=10,
    )
