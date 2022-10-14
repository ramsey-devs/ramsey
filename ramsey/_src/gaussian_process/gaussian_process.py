from typing import Tuple
import haiku as hk
from jax import numpy as jnp

class GP(hk.Module):

    def __init__(self, kernel : hk.Module, x_train : jnp.ndarray, y_train : jnp.ndarray):

        super().__init__()

        self._kernel = kernel
        self._x = x_train
        self._y = y_train

    def __call__(self, method="predict", **kwargs):
        return getattr(self, method)(**kwargs)

    def predict(self, x_s: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        K_tt = self.covariance()
        K_ts = self._kernel(self._x, x_s)
        K_ss = self._kernel(x_s, x_s)

        K_tt_inv = jnp.linalg.inv(K_tt)

        mu_s = K_ts.T.dot(K_tt_inv).dot(self._y)

        cov_s = K_ss - K_ts.T.dot(K_tt_inv).dot(K_ts)
    
        return mu_s, cov_s

    def covariance(self) -> jnp.ndarray:

        sigma_noise = hk.get_parameter("sigma_noise", [], init=hk.initializers.RandomUniform(minval=jnp.log(0.1), maxval=jnp.log(2)))

        K = self._kernel(self._x, self._x) + jnp.exp(sigma_noise)**2  * jnp.eye(len(self._x))
        
        jitter = 1e-6
        K += jitter * jnp.eye(K.shape[0])
        return K

