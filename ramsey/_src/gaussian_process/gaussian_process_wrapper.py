import haiku as hk
import jax
from jax import numpy as jnp
from .kernel import Kernel
from ramsey.models import low_level 

class GP():

    def __init__(self, key : hk.PRNGSequence, kernel : Kernel, x_train : jnp.ndarray, y_train : jnp.ndarray, sigma_noise) -> None:
        self._gp = low_level.GP(key, kernel, x_train, y_train, sigma_noise)

    def train(self, n_iter = 10000, n_restart = 5, init_lr = 1e-3):
        self._gp(method='train', n_iter = n_iter, n_restart = n_restart, init_lr = init_lr)
    
    def predict(self, x_s):
        return self._gp(method='predict', x_s = x_s)
