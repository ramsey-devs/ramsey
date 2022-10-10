import haiku as hk
import jax
from jax import numpy as jnp
from jax.numpy.linalg import inv, det

import optax

class GP(hk.Module):

    def __init__(self, kernel : hk.Module, x_train : jnp.ndarray, y_train : jnp.ndarray, sigma_noise) -> None:

        super().__init__()

        self._kernel = kernel
        self._x = x_train
        self._y = y_train
        self._stddev_noise = sigma_noise**2

    def __call__(self, method="predict", **kwargs):
        return getattr(self, method)(**kwargs)

    def predict(self, x_s):
        K_tt = self._kernel(self._x, self._x) + self._stddev_noise* jnp.eye(len(self._x))
        K_ts = self._kernel(self._x, x_s)
        K_ss = self._kernel(x_s, x_s)

        K_tt_inv = jnp.linalg.inv(K_tt)

        mu_s = K_ts.T.dot(K_tt_inv).dot(self._y)

        cov_s = K_ss - K_ts.T.dot(K_tt_inv).dot(K_ts)
    
        return mu_s, cov_s

    def covariance(self):

        K = self._kernel(self._x, self._x) + self._stddev_noise * jnp.eye(len(self._x))
        return K

